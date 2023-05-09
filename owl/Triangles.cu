// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "Triangles.h"
#include "Context.h"

#if defined(OWL_CAN_DO_DMM) || defined(OWL_CAN_DO_OMM)
#include <optix_micromap.h>
#endif // OWL_CAN_DO_DMM

namespace owl {

	__device__ __inline__ vec2f computeUV(float2 bary, vec2f uv0, vec2f uv1, vec2f uv2)
	{
		return __saturatef(1.0f - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;
	}

#ifdef OWL_CAN_DO_OMM
	// From https://forums.developer.nvidia.com/t/how-to-use-atomiccas-to-implement-atomicadd-short-trouble-adapting-programming-guide-example/22712
	__device__ short atomicOrShort(unsigned short* address, unsigned short val)
	{
		unsigned int* base_address = (unsigned int*)((char*)address - ((size_t)address & 2));
		unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : val;
		unsigned int long_old = atomicOr(base_address, long_val);

		if ((size_t)address & 2) {

			return (unsigned short)(long_old >> 16);

		}
		else {

			unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;

			if (overflow)

				atomicSub(base_address, overflow);

			return (unsigned short)(long_old & 0xffff);

		}
	}

	__global__ void computeOMMArray(
		unsigned short* d_ommIndices,
		vec2f* texCoords,
		vec3i* indexCoords,
		cudaTextureObject_t texturePtr,
		size_t numSubTriangles,
		unsigned int subdivisionLevel)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= numSubTriangles) 
			return;

		int numSubTrianglesPerBaseTriangle = 1 << (subdivisionLevel * 2);

		size_t triIdx = tid / numSubTrianglesPerBaseTriangle;
		size_t uTriI = tid % numSubTrianglesPerBaseTriangle;
		unsigned int bitsPerState = 2;

		// Set micro triangle micro mesh array
		float2 bary0, bary1, bary2;
		optixMicromapIndexToBaseBarycentrics(uTriI, subdivisionLevel, bary0, bary1, bary2);

		vec3i vtxIndices;
		if (indexCoords)
			vtxIndices = indexCoords[triIdx];
		else
			vtxIndices = vec3i(triIdx * 3 + 0, triIdx * 3 + 1, triIdx * 3 + 2);

		const vec2f subTriUV0 = computeUV(bary0, texCoords[vtxIndices.x], texCoords[vtxIndices.y], texCoords[vtxIndices.z]);
		const vec2f subTriUV1 = computeUV(bary1, texCoords[vtxIndices.x], texCoords[vtxIndices.y], texCoords[vtxIndices.z]);
		const vec2f subTriUV2 = computeUV(bary2, texCoords[vtxIndices.x], texCoords[vtxIndices.y], texCoords[vtxIndices.z]);

		float4 textureVal0 = tex2D<float4>(texturePtr, subTriUV0.x, subTriUV0.y);
		float4 textureVal1 = tex2D<float4>(texturePtr, subTriUV1.x, subTriUV1.y);
		float4 textureVal2 = tex2D<float4>(texturePtr, subTriUV2.x, subTriUV2.y);

		float sumAlpha = (textureVal0.w + textureVal1.w + textureVal2.w) / 3.0f;

		int opacity = 1;

		if (sumAlpha < 0.25f)
			opacity = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
		else if (sumAlpha < 0.5f)
			opacity = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
		else if (sumAlpha < 0.75f)
			opacity = OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE;
		else
			opacity = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;

		size_t dataidx = triIdx * (numSubTrianglesPerBaseTriangle / 16 * bitsPerState) + (uTriI / 8);
		atomicOrShort(d_ommIndices + dataidx, (unsigned short)(opacity << (uTriI % 8 * 2)));
	}
#endif // OWL_CAN_DO_OMM

#ifdef OWL_CAN_DO_DMM
	struct DisplacementBlock64MicroTris64B
	{
		// 45 displacement values, implicit vertices
		// 11 bits per displacement values, tightly packed
		// -> 64 bytes per block
		uint8_t data[64];

		// packs the 11 lower bits of the displacement value into the displacement block
		// vertexIdx must be in range [0,44]
		__host__ __device__ __inline__ void setDisplacement(unsigned vertexIdx, uint16_t displacement)
		{
			unsigned int bitWidth = 11;

			unsigned bitOfs = bitWidth * vertexIdx;
			unsigned valueOfs = 0;

			while (valueOfs < bitWidth)
			{
				unsigned num = (~bitOfs & 7) + 1;
				if (bitWidth - valueOfs < num)
					num = bitWidth - valueOfs;

				unsigned mask = (1u << num) - 1u;
				int      idx = bitOfs >> 3;
				int      shift = bitOfs & 7;

				unsigned bits = (unsigned)(displacement >> valueOfs) & mask;  // extract bits from the input value
				data[idx] &= ~(mask << shift);                                // clear bits in memory
				data[idx] |= bits << shift;                                     // insert bits into memory

				valueOfs += num;
				bitOfs += num;
			}
		}
	};

	__global__ void computeDMMArray(
		DisplacementBlock64MicroTris64B* d_displacementBlocks,
		vec3f* d_displacementDirections,
		vec2f* texCoords,
		vec3i* indexCoords,
		vec3f* normals,
		cudaTextureObject_t texturePtr,
		size_t numSubTriangles,
		size_t numTriangles,
		unsigned int dmmSubdivisionLevel,
		const float displacementScale)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= numSubTriangles) return;

		// Offset into vertex index LUT (u major to hierarchical order) for subdivision levels 0 to 3
		// 6  values for subdiv lvl 1
		// 15 values for subdiv lvl 2
		// 45 values for subdiv lvl 3
		const uint16_t UMAJOR_TO_HIERARCHICAL_VTX_IDX_LUT_OFFSET[5] = { 0, 3, 9, 24, 69 };
		// LUTs for levels [0,3]
		const uint16_t UMAJOR_TO_HIERARCHICAL_VTX_IDX_LUT[69] = {
			// level 0
			0, 2, 1,
			// level 1
			0, 3, 2, 5, 4, 1,
			// level 2
			0, 6, 3, 12, 2, 8, 7, 14, 13, 5, 9, 4, 11, 10, 1,
			// level 3
			0, 15, 6, 21, 3, 39, 12, 42, 2, 17, 16, 23, 22, 41, 40, 44, 43, 8, 18, 7, 24, 14, 36, 13, 20, 19, 26, 25, 38, 37, 5, 27, 9, 33, 4, 29, 28, 35, 34, 11, 30, 10, 32, 31, 1 };

		unsigned int numSubTrianglesPerBaseTriangle = numSubTriangles / numTriangles;

		unsigned int triIdx = tid / numSubTrianglesPerBaseTriangle;

		vec3i vtxIndices;
		if (indexCoords)
			vtxIndices = indexCoords[triIdx];
		else
			vtxIndices = vec3i(triIdx * 3 + 0, triIdx * 3 + 1, triIdx * 3 + 2);

		vec2f baseUV0 = texCoords[vtxIndices.x];
		vec2f baseUV1 = texCoords[vtxIndices.y];
		vec2f baseUV2 = texCoords[vtxIndices.z];

		// Set displacement directions per index
		{
			if (normals)
			{
				vec3f normal = normals[vtxIndices.x];
				d_displacementDirections[vtxIndices.x] = normal * displacementScale;

				normal = normals[vtxIndices.y];
				d_displacementDirections[vtxIndices.y] = normal * displacementScale;

				normal = normals[vtxIndices.z];
				d_displacementDirections[vtxIndices.z] = normal * displacementScale;
			}
			else {
				vec3f direction(0.0f, displacementScale, 0.0f);
				d_displacementDirections[vtxIndices.x] = direction;
				d_displacementDirections[vtxIndices.y] = direction;
				d_displacementDirections[vtxIndices.z] = direction;
			}
		}

		// Set micro triangle micro mesh array
		float2 subTriBary0, subTriBary1, subTriBary2;
		unsigned int subTriIdx = tid % numSubTrianglesPerBaseTriangle;
		const unsigned int dmmSubdivisionLevelSubTriangles = max(0, (int)dmmSubdivisionLevel - 3);
		optixMicromapIndexToBaseBarycentrics(subTriIdx, dmmSubdivisionLevelSubTriangles, subTriBary0, subTriBary1, subTriBary2);

		vec2f subTriUV0 = computeUV(subTriBary0, baseUV0, baseUV1, baseUV2);
		vec2f subTriUV1 = computeUV(subTriBary1, baseUV0, baseUV1, baseUV2);
		vec2f subTriUV2 = computeUV(subTriBary2, baseUV0, baseUV1, baseUV2);

		DisplacementBlock64MicroTris64B block;

		unsigned perBlockSubdivisionLevel = min(3u, dmmSubdivisionLevel);
		// fill the displacement block by looping over the vertices in u-major order and use a lookup table to set the corresponding bits in the displacement block
		unsigned numSegments = 1 << perBlockSubdivisionLevel;
		unsigned startVertex = UMAJOR_TO_HIERARCHICAL_VTX_IDX_LUT_OFFSET[perBlockSubdivisionLevel];
		unsigned int uMajorVertIdx = 0;
		for (unsigned iu = 0; iu < numSegments + 1; ++iu)
		{
			for (unsigned iv = 0; iv < numSegments + 1 - iu; ++iv)
			{
				vec2f microVertexBary = { float(iu) / float(numSegments), float(iv) / float(numSegments) };
				vec2f microVertexUV = (1.0f - microVertexBary.x - microVertexBary.y) * subTriUV0 + microVertexBary.x * subTriUV1 + microVertexBary.y * subTriUV2;

				float4 textureVal = tex2D<float4>(texturePtr, microVertexUV.x, microVertexUV.y);
				uint16_t disp = int(__saturatef(textureVal.x) * 0x7FF);

				block.setDisplacement(UMAJOR_TO_HIERARCHICAL_VTX_IDX_LUT[startVertex + uMajorVertIdx], disp);
				uMajorVertIdx++;
			}
		}

		d_displacementBlocks[tid] = block;
	}
#endif // OWL_CAN_DO_DMM


	__device__ __inline__ vec2f computeUV(vec2f bary, vec2f uv0, vec2f uv1, vec2f uv2)
	{
		return (1.0f - bary.x - bary.y) * uv0 + bary.x * uv1 + bary.y * uv2;
	}

	__device__ __inline__ int evaluteOpacity(const vec2f& bary0, const vec2f& bary1, const vec2f& bary2, const vec2f* uvs)
	{
		const vec2f uv0 = computeUV(bary0, uvs[0], uvs[1], uvs[2]);
		const vec2f uv1 = computeUV(bary1, uvs[0], uvs[1], uvs[2]);
		const vec2f uv2 = computeUV(bary2, uvs[0], uvs[1], uvs[2]);

		return 0;
	}

	__device__ static float atomicMax(float* address, float val)
	{
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fmaxf(val, __int_as_float(assumed))));
		} while (assumed != old);
		return __int_as_float(old);
	}

	__device__ static float atomicMin(float* address, float val)
	{
		int* address_as_i = (int*)address;
		int old = *address_as_i, assumed;
		do {
			assumed = old;
			old = ::atomicCAS(address_as_i, assumed,
				__float_as_int(::fminf(val, __int_as_float(assumed))));
		} while (assumed != old);
		return __int_as_float(old);
	}

	/*! device kernel to compute bounding box of vertex array (and thus,
		bbox of triangle mesh, for motion blur (which for instances
		requies knowing the bboxes of its objects */
	__global__ void computeBoundsOfVertices(box3f* d_bounds,
		const void* d_vertices,
		size_t count,
		size_t stride,
		size_t offset)
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= count) return;

		const uint8_t* ptr = (const uint8_t*)d_vertices;
		ptr += tid * stride;
		ptr += offset;

		vec3f vtx = *(const vec3f*)ptr;
		atomicMin(&d_bounds->lower.x, vtx.x);
		atomicMin(&d_bounds->lower.y, vtx.y);
		atomicMin(&d_bounds->lower.z, vtx.z);
		atomicMax(&d_bounds->upper.x, vtx.x);
		atomicMax(&d_bounds->upper.y, vtx.y);
		atomicMax(&d_bounds->upper.z, vtx.z);
	}

	// ------------------------------------------------------------------
	// TrianglesGeomType
	// ------------------------------------------------------------------

	TrianglesGeomType::TrianglesGeomType(Context* const context,
		size_t varStructSize,
		const std::vector<OWLVarDecl>& varDecls)
		: GeomType(context, varStructSize, varDecls)
	{}

	// ------------------------------------------------------------------
	// TrianglesGeomType::createGeom
	// ------------------------------------------------------------------

	std::shared_ptr<Geom> TrianglesGeomType::createGeom()
	{
		GeomType::SP self
			= std::dynamic_pointer_cast<GeomType>(shared_from_this());
		Geom::SP geom = std::make_shared<TrianglesGeom>(context, self);
		geom->createDeviceData(context->getDevices());
		return geom;
	}

	// ------------------------------------------------------------------
	// TrianglesGeom::DeviceData
	// ------------------------------------------------------------------

	TrianglesGeom::DeviceData::DeviceData(const DeviceContext::SP& device)
		: Geom::DeviceData(device)
	{}

	// ------------------------------------------------------------------
	// TrianglesGeom
	// ------------------------------------------------------------------

	TrianglesGeom::TrianglesGeom(Context* const context,
		GeomType::SP geometryType)
		: Geom(context, geometryType)
	{}

	/*! pretty-print */
	std::string TrianglesGeom::toString() const
	{
		return "TrianglesGeom";
	}

	/*! call a cuda kernel that computes the bounds of the vertex buffers */
	void TrianglesGeom::computeBounds(box3f bounds[2])
	{
		assert(vertex.buffers.size() == 1 || vertex.buffers.size() == 2);

		int numThreads = 1024;
		int numBlocks = int((vertex.count + numThreads - 1) / numThreads);

		DeviceContext::SP device = context->getDevice(0);
		assert(device);
		SetActiveGPU forLifeTime(device);

		DeviceMemory d_bounds;
		d_bounds.alloc(2 * sizeof(box3f));
		bounds[0] = bounds[1] = box3f();
		d_bounds.upload(bounds);
		computeBoundsOfVertices << <numBlocks, numThreads >> >
			(((box3f*)d_bounds.get()) + 0,
				vertex.buffers[0]->getPointer(device),
				vertex.count, vertex.stride, vertex.offset);
		if (vertex.buffers.size() == 2)
			computeBoundsOfVertices << <numBlocks, numThreads >> >
			(((box3f*)d_bounds.get()) + 1,
				vertex.buffers[1]->getPointer(device),
				vertex.count, vertex.stride, vertex.offset);
		OWL_CUDA_SYNC_CHECK();
		d_bounds.download(&bounds[0]);
		d_bounds.free();
		OWL_CUDA_SYNC_CHECK();
		if (vertex.buffers.size() == 1)
			bounds[1] = bounds[0];
	}

	/*! call a cuda kernel that computes the Opacity Micro Map */
	void TrianglesGeom::computeOMM(Texture::SP tex)
	{
#ifdef OWL_CAN_DO_OMM
		assert(texCoord.buffer);
		if (subdivisionLevel > 0)
		{
			const unsigned int numSubTrianglesPerBaseTriangle = 1 << (2 * subdivisionLevel);
			unsigned int bitsPerState = 2;

			size_t numTriangles = index.count;
			size_t numSubTriangles = numTriangles * numSubTrianglesPerBaseTriangle;
			size_t numMicroTriangles = numTriangles * numSubTrianglesPerBaseTriangle / 16 * bitsPerState;

			int numThreads = 1024;
			int numBlocks = int((numSubTriangles + numThreads - 1) / numThreads);

			// Calculate omm indices and array
			DeviceContext::SP device = context->getDevice(0);
			assert(device);
			SetActiveGPU forLifeTime(device);

			auto texDD = tex->getObject(device->cudaDeviceID);

			if (texDD)
			{
				DeviceMemory d_ommmIndices;
				std::vector<unsigned short> omm_indices(numMicroTriangles);
				d_ommmIndices.upload(omm_indices);

				auto texCoordsDD = texCoord.buffer->getDD(device);
				auto indexCoordsDD = index.buffer->getDD(device);
				
				computeOMMArray << <numBlocks, numThreads >> > (
					(unsigned short*)d_ommmIndices.d_pointer
					, (vec2f*)texCoordsDD.d_pointer
					, (vec3i*)indexCoordsDD.d_pointer
					, texDD
					, numSubTriangles
					, subdivisionLevel
					);
				OWL_CUDA_SYNC_CHECK();
				
				// Create omm indices per triangle
				d_ommmIndices.download(omm_indices.data());

				// Upload the array and indices to each device
				for (auto device : context->getDevices()) {
					DeviceData& dd = getDD(device);
					dd.ommIndexPointer.upload(omm_indices);
				}
			}
		}
#endif // OWL_CAN_DO_OMM
	}

	/*! call a cuda kernel that computes the displacement micro mesh buffers */
	void TrianglesGeom::computeDMM(Texture::SP tex)
	{
#ifdef OWL_CAN_DO_DMM
		assert(texCoord.buffer);

		if (displacementScale != 0.0f && subdivisionLevel > 0)
		{
			// Based on the subdivision level [0,5], we compute the number of sub triangles.
			// In this sample, we fix the format to OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES, which corresponds to 1 sub triangle at subdivision levels 0-3.
			// Level 4 requires 4 sub triangles, level 5 requires 16 sub triangles.
			const unsigned int dmmSubdivisionLevelSubTriangles = std::max(0, (int)subdivisionLevel - 3);
			const unsigned int numSubTrianglesPerBaseTriangle = 1 << (2 * dmmSubdivisionLevelSubTriangles);

			size_t numTriangles = index.count;
			size_t numSubTriangles = numTriangles * numSubTrianglesPerBaseTriangle;

			DeviceContext::SP device = context->getDevice(0);
			assert(device);
			SetActiveGPU forLifeTime(device);

			auto texDD = tex->getObject(device->cudaDeviceID);

			if (texDD)
			{
				DeviceMemory d_displacementValues;
				DeviceMemory d_displacementDirections;
        
				int numThreads = 1024;
				int numBlocks = int((numSubTriangles + numThreads - 1) / numThreads);

				d_displacementValues.alloc(numSubTriangles * sizeof(DisplacementBlock64MicroTris64B));
				d_displacementDirections.alloc(vertex.count * sizeof(vec3f));

				void* indexCoordsPtr = nullptr;
				void* normalsPtr = nullptr;

				auto texCoordsDD = texCoord.buffer->getDD(device);
				normalsPtr = normal.buffer ? normal.buffer->getDD(device).d_pointer : nullptr;
				indexCoordsPtr = index.buffer ? index.buffer->getDD(device).d_pointer : nullptr;

				computeDMMArray << <numBlocks, numThreads >> > (
					(DisplacementBlock64MicroTris64B*)d_displacementValues.get()
					, (vec3f*)d_displacementDirections.get()
					, (vec2f*)texCoordsDD.d_pointer
					, (vec3i*)indexCoordsPtr
					, (vec3f*)normalsPtr
					, texDD
					, numSubTriangles
					, numTriangles
					, subdivisionLevel
					, displacementScale
					);
				OWL_CUDA_SYNC_CHECK();
				
				std::vector<DisplacementBlock64MicroTris64B> displacementValues(numSubTriangles);
				d_displacementValues.download(displacementValues.data());

				std::vector<vec3f> displacementDirections(vertex.count);
				d_displacementDirections.download(displacementDirections.data());

				for (auto device : context->getDevices())
				{
					DeviceData& dd = getDD(device);
					dd.dmmArray.d_displacementDirections.upload(displacementDirections);
					dd.dmmArray.d_displacementValues.upload(displacementValues);
				}
			}
		}
#endif // OWL_CAN_DO_DMM
	}

	RegisteredObject::DeviceData::SP TrianglesGeom::createOn(const DeviceContext::SP& device)
	{
		return std::make_shared<DeviceData>(device);
	}

	/*! set the vertex array (if vector size is 1), or set/enable
	  motion blur via multiple time steps, if vector size >= 0 */
	void TrianglesGeom::setVertices(const std::vector<Buffer::SP>& vertexArrays,
		/*! the number of vertices in each time step */
		size_t count,
		size_t stride,
		size_t offset)
	{
		vertex.buffers = vertexArrays;
		vertex.count = count;
		vertex.stride = stride;
		vertex.offset = offset;

		for (auto device : context->getDevices()) {
			DeviceData& dd = getDD(device);
			dd.vertexPointers.clear();
			for (auto va : vertexArrays)
				dd.vertexPointers.push_back((CUdeviceptr)va->getPointer(device) + offset);
		}
	}

	void TrianglesGeom::setIndices(Buffer::SP indices,
		size_t count,
		size_t stride,
		size_t offset)
	{
		index.buffer = indices;
		index.count = count;
		index.stride = stride;
		index.offset = offset;

		for (auto device : context->getDevices()) {
			DeviceData& dd = getDD(device);
			dd.indexPointer = (CUdeviceptr)indices->getPointer(device) + offset;
		}
	}

	void TrianglesGeom::setNormals(Buffer::SP normals,
		size_t count,
		size_t stride,
		size_t offset)
	{
		normal.buffer = normals;
		normal.count = count;
		normal.stride = stride;
		normal.offset = offset;

		for (auto device : context->getDevices()) {
			DeviceData& dd = getDD(device);
			dd.normalPointer = (CUdeviceptr)normals->getPointer(device) + offset;
		}
	}

	void TrianglesGeom::setTexCoord(Buffer::SP texCoords,
		size_t count,
		size_t stride,
		size_t offset)
	{
		texCoord.buffer = texCoords;
		texCoord.count = count;
		texCoord.stride = stride;
		texCoord.offset = offset;

		for (auto device : context->getDevices()) {
			DeviceData& dd = getDD(device);
			dd.texCoordPointer = (CUdeviceptr)texCoords->getPointer(device) + offset;
		}
	}
} // ::owl