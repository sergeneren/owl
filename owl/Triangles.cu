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

#ifdef OWL_CAN_DO_DMM
#include <optix_micromap.h>
#endif // OWL_CAN_DO_DMM

namespace owl {

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
		vec3f* normals,
		cudaTextureObject_t texturePtr,
		unsigned int numSubTriangles,
		unsigned int numTriangles,
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

		static const int SEGMENT_TO_MAJOR_VERT_IDX[9][9] = {
				{0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8  },
				{9 , 10, 11, 12, 13, 14, 15, 16, -1 },
				{17, 18, 19, 20, 21, 22, 23, -1, -1 },
				{24, 25, 26, 27, 28, 29, -1, -1, -1 },
				{30, 31, 32, 33, 34, -1, -1, -1, -1 },
				{35, 36, 37, 38, -1, -1, -1, -1, -1 },
				{39, 40, 41, -1, -1, -1, -1, -1, -1 },
				{42, 43, -1, -1, -1, -1, -1, -1, -1 },
				{44, -1, -1, -1, -1, -1, -1, -1, -1 }
			};

		unsigned int numSubTrianglesPerBaseTriangle = numSubTriangles / numTriangles;

		unsigned int triIdx = tid / numSubTrianglesPerBaseTriangle;

		unsigned int vtxIdx0 = triIdx * 3 + 0;
		unsigned int vtxIdx1 = triIdx * 3 + 1;
		unsigned int vtxIdx2 = triIdx * 3 + 2;

		vec2f baseUV0 = texCoords[vtxIdx0];
		vec2f baseUV1 = texCoords[vtxIdx1];
		vec2f baseUV2 = texCoords[vtxIdx2];

		// Set displacement directions per index
		{
			if (normals)
			{
				vec3f normal = normals[vtxIdx0];
				d_displacementDirections[vtxIdx0] = normal * displacementScale;

				normal = normals[vtxIdx1];
				d_displacementDirections[vtxIdx1] = normal * displacementScale;

				normal = normals[vtxIdx2];
				d_displacementDirections[vtxIdx2] = normal * displacementScale;
			}
			else {
				vec3f direction(0.0f, displacementScale, 0.0f);
				d_displacementDirections[vtxIdx0] = direction;
				d_displacementDirections[vtxIdx1] = direction;
				d_displacementDirections[vtxIdx2] = direction;
			}
		}

		// Set micro triangle micro mesh array
		float2 subTriBary0, subTriBary1, subTriBary2;
		unsigned int subTriIdx = tid % numSubTrianglesPerBaseTriangle;
		const unsigned int dmmSubdivisionLevelSubTriangles = max(0, (int)dmmSubdivisionLevel - 3);
		optixMicromapIndexToBaseBarycentrics(subTriIdx, dmmSubdivisionLevelSubTriangles, subTriBary0, subTriBary1, subTriBary2);

		vec2f subTriUV0 = (1.0f - subTriBary0.x - subTriBary0.y) * baseUV0 + subTriBary0.x * baseUV1 + subTriBary0.y * baseUV2;
		vec2f subTriUV1 = (1.0f - subTriBary1.x - subTriBary1.y) * baseUV0 + subTriBary1.x * baseUV1 + subTriBary1.y * baseUV2;
		vec2f subTriUV2 = (1.0f - subTriBary2.x - subTriBary2.y) * baseUV0 + subTriBary2.x * baseUV1 + subTriBary2.y * baseUV2;

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

				uMajorVertIdx = SEGMENT_TO_MAJOR_VERT_IDX[iu][iv];
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

	/*! device kernel to compute bounding box of vertex array (and thus,
	  bbox of triangle mesh, for motion blur (which for instances
	  requies knowing the bboxes of its objects */
	__global__ void computeOpacityArray(const void* d_texCoords,
		unsigned short* omm_input_data,
		cudaTextureObject_t texturePtr,
		unsigned int subdivision_level,
		size_t count, // Number of triangles * micromesh
		size_t stride) // number of micromesh
	{
		int tid = blockDim.x * blockIdx.x + threadIdx.x;
		if (tid >= count) return;

		/*
		unsigned int numMicroTris = 1 << (subdivision_level);

		for( uint32_t uTriI = 0; uTriI < numMicroTris; ++uTriI )
		{
		  vec2f bary0, bary1, bary2;
		  micro2bary( uTriI, subdivision_level, bary0, bary1, bary2 );

		  // first triangle (a,b,c)
		  {
			  const int opacity = evaluteOpacity( bary0, bary1, bary2, g_uvs[0] );
			  omm_input_data[0][uTriI/8] |= opacity << ( uTriI%8 * 2 );
		  }

		  // second triangle (a,c,d)
		  {
			  const int opacity = evaluteOpacity( bary0, bary1, bary2, g_uvs[1] );
			  omm_input_data[1][uTriI/8] |= opacity << ( uTriI%8 * 2 );
		  }
		}

		const uint8_t *texCoordPtr = (const uint8_t *)d_texCoords;
		texCoordPtr += tid;

		vec2f vtx = *(const vec2f*)texCoordPtr;

		float4 texValue = tex2D<float4>(texturePtr,vtx.x, vtx.y);

		unsigned int microTriIndex = 0;

		vec2f bary0, bary1, bary2;
		micro2bary( microTriIndex, subdivision_level, bary0, bary1, bary2 );

		const int opacity = evaluteOpacity( bary0, bary1, bary2, g_uvs[0] );
		omm_input_data[tid] |= opacity << ( uTriI%8 * 2 );
		*/
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
	void TrianglesGeom::computeOMM(Texture& tex)
	{
#ifdef OWL_CAN_DO_OMM
		assert(texCoord.buffer);
		if (subdivisionLevel > 0)
		{
			unsigned int NUM_MICRO_TRIS = 1 << (subdivisionLevel * 2);
			unsigned int BITS_PER_STATE = 2;
			size_t NUM_TRIS = index.count / 3;

			int numThreads = 1024;
			int numBlocks = int((texCoord.count + numThreads - 1) / numThreads);

			// Calculate omm indices and array
			DeviceContext::SP device = context->getDevice(0);
			assert(device);
			SetActiveGPU forLifeTime(device);

			auto texDD = tex.getObject(device->cudaDeviceID);

			// Create omm indices per triangle
			std::vector<unsigned int> omm_indices(NUM_TRIS);
			for (unsigned int i = 0; i < NUM_TRIS; i++)
				omm_indices.push_back(i);

			const size_t omm_indices_size_bytes = omm_indices.size() * sizeof(unsigned int);

			// Upload the array and indices to each device
			for (auto device : context->getDevices()) {
				DeviceData& dd = getDD(device);

				DeviceMemory d_omm_indices;
				d_omm_indices.upload<unsigned int>(omm_indices);
				dd.ommIndexPointer = d_omm_indices.d_pointer;
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
			constexpr int      subTriSizeByteSize = 64;  // 64B for format OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES

			size_t numTriangles = index.count;
			size_t numSubTriangles = numTriangles * numSubTrianglesPerBaseTriangle;

			DeviceContext::SP device = context->getDevice(0);
			assert(device);
			SetActiveGPU forLifeTime(device);

			auto texDD = tex->getObject(device->cudaDeviceID);

			if (texDD)
			{
					DeviceData& dd = getDD(device);

					DeviceMemory d_displacementValues;
					DeviceMemory d_displacementDirections;
        
					int numThreads = 256;
					int numBlocks = int((numSubTriangles + numThreads - 1) / numThreads);

					d_displacementValues.alloc(numSubTriangles * sizeof(DisplacementBlock64MicroTris64B));
					d_displacementDirections.alloc(vertex.count * sizeof(vec3f));

					auto texCoordsDD = texCoord.buffer->getDD(device);
					auto normalsDD = normal.buffer->getDD(device);

					computeDMMArray << <numBlocks, numThreads >> > (
						(DisplacementBlock64MicroTris64B*)d_displacementValues.get()
						, (vec3f*)d_displacementDirections.get()
						, (vec2f*)texCoordsDD.d_pointer
						, (vec3f*)normalsDD.d_pointer
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