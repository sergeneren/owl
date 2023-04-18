// ======================================================================== //
// Copyright 2019-2023 Ingo Wald                                            //
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

#include "TrianglesGeomGroup.h"
#include "Triangles.h"
#include "Context.h"

#define LOG(message)                                            \
  if (Context::logging())                                       \
    std::cout << "#owl(" << device->ID << "): "                 \
              << message                                        \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl(" << device->ID << "): "                 \
              << message << OWL_TERMINAL_DEFAULT << std::endl


namespace owl {

  /*! pretty-printer, for printf-debugging */
  std::string TrianglesGeomGroup::toString() const
  {
    return "TrianglesGeomGroup";
  }
  
  /*! constructor - mostly passthrough to parent class */
  TrianglesGeomGroup::TrianglesGeomGroup(Context *const context,
                                         size_t numChildren,
                                         unsigned int _buildFlags)
    : GeomGroup(context,numChildren), 
    buildFlags( (_buildFlags > 0) ? _buildFlags : defaultBuildFlags)
  {
  }
  
  void TrianglesGeomGroup::updateMotionBounds()
  {
    bounds[0] = bounds[1] = box3f();
    for (auto geom : geometries) {
      TrianglesGeom::SP mesh = geom->as<TrianglesGeom>();
      box3f meshBounds[2];
      mesh->computeBounds(meshBounds);
      for (int i=0;i<2;i++)
        bounds[i].extend(meshBounds[i]);
    }
  }
  
  void TrianglesGeomGroup::buildAccel()
  {
    for (auto device : context->getDevices()) 
      buildAccelOn<true>(device);

    if (context->motionBlurEnabled)
      updateMotionBounds();
  }
  
  void TrianglesGeomGroup::refitAccel()
  {
    for (auto device : context->getDevices()) 
      buildAccelOn<false>(device);
    
    if (context->motionBlurEnabled)
      updateMotionBounds();
  }
  
  template<bool FULL_REBUILD>
  void TrianglesGeomGroup::buildAccelOn(const DeviceContext::SP &device) 
  {
    SetActiveGPU forLifeTime(device);
    DeviceData &dd = getDD(device);

    if (FULL_REBUILD && !dd.bvhMemory.empty())
      dd.bvhMemory.free();

    if (!FULL_REBUILD && dd.bvhMemory.empty())
      throw std::runtime_error("trying to refit an accel struct that has not been previously built");

    if (!FULL_REBUILD && !(buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE))
      throw std::runtime_error("trying to refit an accel struct that was not built with OPTIX_BUILD_FLAG_ALLOW_UPDATE");

    if (FULL_REBUILD) {
      dd.memFinal = 0;
      dd.memPeak = 0;
    }
   
    LOG("building triangles accel over "
        << geometries.size() << " geometries");
    size_t   sumPrims = 0;
    uint32_t maxPrimsPerGAS = 0;
    optixDeviceContextGetProperty
      (device->optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
       &maxPrimsPerGAS,
       sizeof(maxPrimsPerGAS));

    assert(!geometries.empty());
    TrianglesGeom::SP child0 = geometries[0]->as<TrianglesGeom>();
    assert(child0);
    int numKeys = (int)child0->vertex.buffers.size();
    assert(numKeys > 0);
    const bool hasMotion = (numKeys > 1);
    if (hasMotion) assert(context->motionBlurEnabled);
    
    // ==================================================================
    // create triangle inputs
    // ==================================================================
    //! the N build inputs that go into the builder
    std::vector<OptixBuildInput> triangleInputs(geometries.size());
    // one build flag per build input
    std::vector<uint32_t> triangleInputFlags(geometries.size());

    size_t maxNumbytesDmm = 0;

    // now go over all geometries to set up the buildinputs
    for (size_t childID=0;childID<geometries.size();childID++) {

      // the child wer're setting them with (with sanity checks)
      TrianglesGeom::SP tris = geometries[childID]->as<TrianglesGeom>();
      assert(tris);

      if (tris->vertex.buffers.size() != (size_t)numKeys)
        OWL_RAISE("invalid combination of meshes with "
                  "different motion keys in the same "
                  "triangles geom group");
      TrianglesGeom::DeviceData &trisDD = tris->getDD(device);

#ifdef OWL_CAN_DO_OMM
      // TODO build deviceData omm device pointers 
      OptixBuildInputOpacityMicromap ommInput = {};
      if(trisDD.ommArrayPointer && trisDD.ommIndexPointer && tris->subdivisionLevel > 0)
      {
          // Build Opacity Micro Map
          unsigned int numTris =  (unsigned int)(tris->index.count); // Number of triangles 
	      OptixOpacityMicromapUsageCount usage_count = {};
          usage_count.count = numTris;
	      usage_count.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
	      usage_count.subdivisionLevel = tris->subdivisionLevel;
               
	      ommInput.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_INDEXED;
	      ommInput.opacityMicromapArray = trisDD.ommArrayPointer;
	      ommInput.indexBuffer = trisDD.ommIndexPointer;
	      ommInput.indexSizeInBytes = 2;
	      ommInput.numMicromapUsageCounts = 1;
	      ommInput.micromapUsageCounts = &usage_count;
      }
#endif // OWL_CAN_DO_OMM
      
      CUdeviceptr     *d_vertices    = trisDD.vertexPointers.data();
      assert(d_vertices);
      OptixBuildInput &triangleInput = triangleInputs[childID];
      
      triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
      auto &ta = triangleInput.triangleArray;
      ta.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
      ta.vertexStrideInBytes = (uint32_t)tris->vertex.stride;
      ta.numVertices         = (uint32_t)tris->vertex.count;
      ta.vertexBuffers       = d_vertices;
      
      ta.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
      ta.indexStrideInBytes  = (uint32_t)tris->index.stride;
      ta.numIndexTriplets    = (uint32_t)tris->index.count;
      ta.indexBuffer         = trisDD.indexPointer;

#ifdef OWL_CAN_DO_OMM
      ta.opacityMicromap     = ommInput;
#endif // OWL_CAN_DO_OMM

#ifdef OWL_CAN_DO_DMM
      auto &dmm  = trisDD.dmmArray;
      if(tris->subdivisionLevel>0 && tris->displacementScale!=0.0f && dmm.d_displacementValues && dmm.d_displacementDirections)
      {
		  unsigned int dmmSubdivisionLevelSubTriangles = std::max(0, (int)tris->subdivisionLevel - 3);
		  unsigned int numSubTrianglesPerBaseTriangle = 1 << (2 * dmmSubdivisionLevelSubTriangles);
		  constexpr int      subTriSizeByteSize = 64;  // 64B for format OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES

		  size_t numTriangles = tris->index.count;
		  size_t numSubTriangles = numTriangles * numSubTrianglesPerBaseTriangle;

		  //////////////////////////////////////////////////////////////////////////
		  // The actual build of the displacement micromap array.
		  // Only the displacement values are needed here along with the descriptors.
		  // How these values are applied to triangles (displacement directions, indexing, potential scale/bias) is specified at the triangle build input (GAS build)         
		  OptixDisplacementMicromapArrayBuildInput bi = {};

		  bi.flags = OPTIX_DISPLACEMENT_MICROMAP_FLAG_NONE;
		  // We have a very simple distribution of subdivision levels and format usage.
		  // All triangles of the mesh use the uncompressed format, and a fixed subdivision level.
		  // As such, the histogram over the different formats/subdivision levels has only a single entry.
		  // Also, none of the displacement micromaps are re-used between triangles, so we put 'numTriangles' displacement micromaps into an array.

          OptixDisplacementMicromapHistogramEntry histogram;

		  histogram.count = (unsigned int)numTriangles;
		  histogram.format = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
		  histogram.subdivisionLevel = tris->subdivisionLevel;
		  
          bi.numDisplacementMicromapHistogramEntries = 1;
		  bi.displacementMicromapHistogramEntries = &histogram;

		  OptixMicromapBufferSizes bs = {};
		  OPTIX_CHECK(optixDisplacementMicromapArrayComputeMemoryUsage(device->optixContext, &bi, &bs));

		  // Provide the device data for the DMM array build
		  std::vector<OptixDisplacementMicromapDesc> descriptors(numTriangles);
		  for (unsigned int i = 0; i < numTriangles; ++i)
		  {
			  OptixDisplacementMicromapDesc& desc = descriptors[i];
			  desc.byteOffset = i * subTriSizeByteSize * numSubTrianglesPerBaseTriangle;
			  desc.format = OPTIX_DISPLACEMENT_MICROMAP_FORMAT_64_MICRO_TRIS_64_BYTES;
			  desc.subdivisionLevel = tris->subdivisionLevel;
		  }

		  DeviceMemory d_descriptors;
		  d_descriptors.upload(descriptors);

		  bi.perDisplacementMicromapDescBuffer = d_descriptors.d_pointer;
		  bi.displacementValuesBuffer = trisDD.dmmArray.d_displacementValues;

		  dmm.d_dmmArrayData.alloc(bs.outputSizeInBytes);
		  dmm.d_build_temp.alloc(bs.outputSizeInBytes);

          maxNumbytesDmm = max(maxNumbytesDmm, bs.outputSizeInBytes);

		  OptixMicromapBuffers uBuffers = {};
		  uBuffers.output = dmm.d_dmmArrayData.d_pointer;
		  uBuffers.outputSizeInBytes = dmm.d_dmmArrayData.sizeInBytes;
		  uBuffers.temp = dmm.d_build_temp.d_pointer;
		  uBuffers.tempSizeInBytes = dmm.d_build_temp.sizeInBytes;

		  OPTIX_CHECK(optixDisplacementMicromapArrayBuild(device->optixContext, /* todo: stream */0, &bi, &uBuffers));

          OptixDisplacementMicromapUsageCount usage = {};
	      OptixBuildInputDisplacementMicromap& disp = ta.displacementMicromap;

	      disp.indexingMode = OPTIX_DISPLACEMENT_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
	      disp.displacementMicromapArray = dmm.d_dmmArrayData.d_pointer;

	      // Displacement directions, 3 vectors (these do not need to be normalized!)
	      // While the API accepts float values for convenience, OptiX uses the half format internally. Float inputs are converted to half.
	      // So it is usually best to input half values directly to control precision.
	      disp.vertexDirectionsBuffer = dmm.d_displacementDirections;
	      disp.vertexDirectionFormat = OPTIX_DISPLACEMENT_MICROMAP_DIRECTION_FORMAT_FLOAT3;

	      // Since we create exactly one displacement micromap per triangle and we apply a displacement micromap to every triangle, there
	      // is a one to one mapping between the DMM usage and the DMM histogram
	      // we could even do a reinterpret_cast from dmm histogram to dmm usage here
	      usage.count = histogram.count;
	      usage.format = histogram.format;
	      usage.subdivisionLevel = histogram.subdivisionLevel;
	      disp.numDisplacementMicromapUsageCounts = 1;
	      disp.displacementMicromapUsageCounts = &usage;
      }
#endif // OWL_CAN_DO_DMM

      assert(ta.indexBuffer);
      
      // -------------------------------------------------------
      // sanity check that we don't have too many prims
      // -------------------------------------------------------
      sumPrims += ta.numIndexTriplets;
      // we always have exactly one SBT entry per shape (i.e., triangle
      // mesh), and no per-primitive materials:
      triangleInputFlags[childID]    = OPTIX_GEOMETRY_FLAG_REQUIRE_SINGLE_ANYHIT_CALL;//0;
      ta.flags                       = &triangleInputFlags[childID];
      // iw, jan 7, 2020: note this is not the "actual" number of
      // SBT entires we'll generate when we build the SBT, only the
      // number of per-ray-type 'groups' of SBT entities (i.e., before
      // scaling by the SBT_STRIDE that gets passed to
      // optixTrace. So, for the build input this value remains *1*).
      ta.numSbtRecords               = 1; 
      ta.sbtIndexOffsetBuffer        = 0; 
      ta.sbtIndexOffsetSizeInBytes   = 0; 
      ta.sbtIndexOffsetStrideInBytes = 0; 
    }
    
    if (sumPrims > maxPrimsPerGAS) 
      OWL_RAISE("number of prim in user geom group exceeds "
                "OptiX's MAX_PRIMITIVES_PER_GAS limit");
    
    // ==================================================================
    // BLAS setup: buildinputs set up, build the blas
    // ==================================================================
      
    // ------------------------------------------------------------------
    // first: compute temp memory for bvh
    // ------------------------------------------------------------------
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = this->buildFlags;

#ifdef OWL_CAN_DO_DMM
    accelOptions.buildFlags |=  OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE ;
#endif // OWL_CAN_DO_DMM
    
    accelOptions.motionOptions.numKeys   = numKeys;
    accelOptions.motionOptions.flags     = 0;
    accelOptions.motionOptions.timeBegin = 0.f;
    accelOptions.motionOptions.timeEnd   = 1.f;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
      
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (device->optixContext,
                 &accelOptions,
                 triangleInputs.data(),
                 (uint32_t)triangleInputs.size(),
                 &blasBufferSizes
                 ));
    
    // ------------------------------------------------------------------
    // ... and allocate buffers: temp buffer, initial (uncompacted)
    // BVH buffer, and a one-single-size_t buffer to store the
    // compacted size in
    // ------------------------------------------------------------------

    const size_t tempSize
      = FULL_REBUILD
      ? max(blasBufferSizes.tempSizeInBytes,blasBufferSizes.tempUpdateSizeInBytes)
      : blasBufferSizes.tempUpdateSizeInBytes;
    LOG("starting to build/refit "
        << prettyNumber(triangleInputs.size()) << " triangle geom groups, "
        << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
        << prettyNumber(tempSize) << "B in temp data");

    // temp memory:
    DeviceMemory tempBuffer;
    if (Context::useManagedMemForAccelAux)
      tempBuffer.allocManaged(FULL_REBUILD
                       ?max(blasBufferSizes.tempSizeInBytes,
                            blasBufferSizes.tempUpdateSizeInBytes)
                       :blasBufferSizes.tempUpdateSizeInBytes);
    else
      tempBuffer.alloc(FULL_REBUILD
                       ?max(maxNumbytesDmm, max(blasBufferSizes.tempSizeInBytes,
                            blasBufferSizes.tempUpdateSizeInBytes))
                       :blasBufferSizes.tempUpdateSizeInBytes);
    if (FULL_REBUILD) {
      // Only track this on first build, assuming temp buffer gets smaller for refit
      dd.memPeak += tempBuffer.size();
    }
         
    const bool allowCompaction = (buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION);

    // Optional buffers only used when compaction is allowed
    DeviceMemory outputBuffer;
    DeviceMemory compactedSizeBuffer;


    // Allocate output buffer for initial build
    if (FULL_REBUILD) {
      if (allowCompaction) {
        if (Context::useManagedMemForAccelData)
          outputBuffer.allocManaged(blasBufferSizes.outputSizeInBytes);
        else
          outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
        dd.memPeak += outputBuffer.size();
      } else {
        if (Context::useManagedMemForAccelData)
          dd.bvhMemory.allocManaged(blasBufferSizes.outputSizeInBytes);
        else
          dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
        dd.memPeak += dd.bvhMemory.size();
        dd.memFinal = dd.bvhMemory.size();
      }
    }

    // Build or refit

    if (FULL_REBUILD && allowCompaction) {

      if (Context::useManagedMemForAccelData)
        compactedSizeBuffer.allocManaged(sizeof(uint64_t));
      else
        compactedSizeBuffer.alloc(sizeof(uint64_t));
      dd.memPeak += compactedSizeBuffer.size();

      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = (CUdeviceptr)compactedSizeBuffer.get();

      // Initial, uncompacted build
      OPTIX_CHECK(optixAccelBuild(device->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  triangleInputs.data(),
                                  (uint32_t)triangleInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)outputBuffer.get(),
                                  outputBuffer.size(),
                                  /* the traversable we're building: */ 
                                  &dd.traversable,
                                  /* we're also querying compacted size: */
                                  &emitDesc,1u
                                  ));
      OWL_CUDA_SYNC_CHECK();
    } else {

      // This is either a full rebuild operation _without_ compaction, or a refit.
      // The operation has already been stored in accelOptions.

      OPTIX_CHECK(optixAccelBuild(device->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  triangleInputs.data(),
                                  (uint32_t)triangleInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)dd.bvhMemory.get(),
                                  dd.bvhMemory.size(),
                                  /* the traversable we're building: */ 
                                  &dd.traversable,
                                  /* we're also querying compacted size: */
                                  nullptr,0
                                  ));
    OWL_CUDA_SYNC_CHECK();
    }
    
    // ==================================================================
    // perform compaction
    // ==================================================================
    
    if (FULL_REBUILD && allowCompaction) {
      // download builder's compacted size from device
      uint64_t compactedSize;
      compactedSizeBuffer.download(&compactedSize);

      if (Context::useManagedMemForAccelData)
        dd.bvhMemory.allocManaged(compactedSize);
      else
        dd.bvhMemory.alloc(compactedSize);
      // ... and perform compaction
      OPTIX_CALL(AccelCompact(device->optixContext,
                              /*TODO: stream:*/0,
                              // OPTIX_COPY_MODE_COMPACT,
                              dd.traversable,
                              (CUdeviceptr)dd.bvhMemory.get(),
                              dd.bvhMemory.size(),
                              &dd.traversable));
      dd.memPeak += dd.bvhMemory.size();
      dd.memFinal = dd.bvhMemory.size();
    }
    OWL_CUDA_SYNC_CHECK();
      
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    if (FULL_REBUILD)
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    if (FULL_REBUILD)
      compactedSizeBuffer.free();
    
    LOG_OK("successfully build triangles geom group accel");
  }
  
} // ::owl
