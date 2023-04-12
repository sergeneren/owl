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

namespace owl {

  // Deinterleave bits from x into even and odd halves
  __device__ __inline__ uint32_t deinterleaveBits( uint32_t x )
  {
      x = ( ( ( ( x >> 1 ) & 0x22222222u ) | ( ( x << 1 ) & ~0x22222222u ) ) & 0x66666666u ) | ( x & ~0x66666666u );
      x = ( ( ( ( x >> 2 ) & 0x0c0c0c0cu ) | ( ( x << 2 ) & ~0x0c0c0c0cu ) ) & 0x3c3c3c3cu ) | ( x & ~0x3c3c3c3cu );
      x = ( ( ( ( x >> 4 ) & 0x00f000f0u ) | ( ( x << 4 ) & ~0x00f000f0u ) ) & 0x0ff00ff0u ) | ( x & ~0x0ff00ff0u );
      x = ( ( ( ( x >> 8 ) & 0x0000ff00u ) | ( ( x << 8 ) & ~0x0000ff00u ) ) & 0x00ffff00u ) | ( x & ~0x00ffff00u );
      return x;
  }
  
  // Extract even bits
  __device__ __inline__ uint32_t extractEvenBits( uint32_t x )
  {
      x &= 0x55555555;
      x = ( x | ( x >> 1 ) ) & 0x33333333;
      x = ( x | ( x >> 2 ) ) & 0x0f0f0f0f;
      x = ( x | ( x >> 4 ) ) & 0x00ff00ff;
      x = ( x | ( x >> 8 ) ) & 0x0000ffff;
      return x;
  }
  
  
  // Calculate exclusive prefix or (log(n) XOR's and SHF's)
  __device__ __inline__ uint32_t prefixEor( uint32_t x )
  {
      x ^= x >> 1;
      x ^= x >> 2;
      x ^= x >> 4;
      x ^= x >> 8;
      return x;
  }
  
  
  // Convert distance along the curve to discrete barycentrics
  __device__ __inline__ void index2dbary( uint32_t index, uint32_t& u, uint32_t& v, uint32_t& w )
  {
      uint32_t b0 = extractEvenBits( index );
      uint32_t b1 = extractEvenBits( index >> 1 );
  
      uint32_t fx = prefixEor( b0 );
      uint32_t fy = prefixEor( b0 & ~b1 );
  
      uint32_t t = fy ^ b1;
  
      u = ( fx & ~t ) | ( b0 & ~t ) | ( ~b0 & ~fx & t );
      v = fy ^ b0;
      w = ( ~fx & ~t ) | ( b0 & ~t ) | ( ~b0 & fx & t );
  }
  
  
  // Compute barycentrics for micro triangle
  __device__ __inline__ void micro2bary( uint32_t index, uint32_t subdivisionLevel, vec2f& uv0, vec2f& uv1, vec2f& uv2 )
  {
      if( subdivisionLevel == 0 )
      {
          uv0 = { 0, 0 };
          uv1 = { 1, 0 };
          uv2 = { 0, 1 };
          return;
      }
  
      uint32_t iu, iv, iw;
      index2dbary( index, iu, iv, iw );
  
      // we need to only look at "level" bits
      iu = iu & ( ( 1 << subdivisionLevel ) - 1 );
      iv = iv & ( ( 1 << subdivisionLevel ) - 1 );
      iw = iw & ( ( 1 << subdivisionLevel ) - 1 );
  
      bool upright = ( iu & 1 ) ^ ( iv & 1 ) ^ ( iw & 1 );
      if( !upright )
      {
          iu = iu + 1;
          iv = iv + 1;
      }
  
      const float levelScale = __uint_as_float( ( 127u - subdivisionLevel ) << 23 );
  
      // scale the barycentic coordinate to the global space/scale
      float du = 1.f * levelScale;
      float dv = 1.f * levelScale;
  
      // scale the barycentic coordinate to the global space/scale
      float u = (float)iu * levelScale;
      float v = (float)iv * levelScale;
  
      if( !upright )
      {
          du = -du;
          dv = -dv;
      }
  
      uv0 = { u, v };
      uv1 = { u + du, v };
      uv2 = { u, v + dv };
  }

  __device__ __inline__ vec2f computeUV( vec2f bary, vec2f uv0, vec2f uv1, vec2f uv2 )
  {
      return ( 1.0f - bary.x - bary.y )*uv0 + bary.x*uv1 + bary.y*uv2;
  }

  __device__ __inline__ int evaluteOpacity()
  {
     const vec2f uv0 = computeUV(bary0, uvs[0], uvs[1], uvs[2] );
     const vec2f uv1 = computeUV(bary1, uvs[0], uvs[1], uvs[2] );
     const vec2f uv2 = computeUV(bary2, uvs[0], uvs[1], uvs[2] );
  
  }

  __device__ static float atomicMax(float* address, float val)
  {
    int* address_as_i = (int*) address;
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
    int* address_as_i = (int*) address;
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
  __global__ void computeBoundsOfVertices(box3f *d_bounds,
                                          const void *d_vertices,
                                          size_t count,
                                          size_t stride,
                                          size_t offset)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= count) return;

    const uint8_t *ptr = (const uint8_t *)d_vertices;
    ptr += tid*stride;
    ptr += offset;

    vec3f vtx = *(const vec3f*)ptr;
    atomicMin(&d_bounds->lower.x,vtx.x);
    atomicMin(&d_bounds->lower.y,vtx.y);
    atomicMin(&d_bounds->lower.z,vtx.z);
    atomicMax(&d_bounds->upper.x,vtx.x);
    atomicMax(&d_bounds->upper.y,vtx.y);
    atomicMax(&d_bounds->upper.z,vtx.z);
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

  }
                                          
  // ------------------------------------------------------------------
  // TrianglesGeomType
  // ------------------------------------------------------------------
  
  TrianglesGeomType::TrianglesGeomType(Context *const context,
                                       size_t varStructSize,
                                       const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls)
  {}

  // ------------------------------------------------------------------
  // TrianglesGeomType::createGeom
  // ------------------------------------------------------------------
  
  std::shared_ptr<Geom> TrianglesGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<TrianglesGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  
  // ------------------------------------------------------------------
  // TrianglesGeom::DeviceData
  // ------------------------------------------------------------------
  
  TrianglesGeom::DeviceData::DeviceData(const DeviceContext::SP &device)
    : Geom::DeviceData(device)
  {}
  
  
  // ------------------------------------------------------------------
  // TrianglesGeom
  // ------------------------------------------------------------------
  
  TrianglesGeom::TrianglesGeom(Context *const context,
                               GeomType::SP geometryType)
    : Geom(context,geometryType)
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
    d_bounds.alloc(2*sizeof(box3f));
    bounds[0] = bounds[1] = box3f();
    d_bounds.upload(bounds);
    computeBoundsOfVertices<<<numBlocks,numThreads>>>
      (((box3f*)d_bounds.get())+0,
       vertex.buffers[0]->getPointer(device),
       vertex.count,vertex.stride,vertex.offset);
    if (vertex.buffers.size() == 2)
      computeBoundsOfVertices<<<numBlocks,numThreads>>>
        (((box3f*)d_bounds.get())+1,
         vertex.buffers[1]->getPointer(device),
         vertex.count,vertex.stride,vertex.offset);
    OWL_CUDA_SYNC_CHECK();
    d_bounds.download(&bounds[0]);
    d_bounds.free();
    OWL_CUDA_SYNC_CHECK();
    if (vertex.buffers.size() == 1)
      bounds[1] = bounds[0];
  }
  
  /*! call a cuda kernel that computes the bounds of the vertex buffers */
  void TrianglesGeom::computeOMM(Texture& tex)
  {
    assert(texCoord.buffers.size());
    assert(subdivisionLevel > 0);
   
    unsigned int NUM_MICRO_TRIS = 1 << ( subdivisionLevel*2 );
    unsigned int BITS_PER_STATE = 2;
    unsigned int NUM_TRIS = index.count / 3;

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

    const size_t omm_indices_size_bytes = omm_indices.size() * sizeof( unsigned int ); 

    // Upload the array and indices to each device  
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      
      DeviceMemory d_omm_indices;
      d_omm_indices.upload<unsigned int>(omm_indices);
      dd.ommIndexPointer = d_omm_indices.d_pointer;
    }
  }

  RegisteredObject::DeviceData::SP TrianglesGeom::createOn(const DeviceContext::SP &device) 
  { return std::make_shared<DeviceData>(device); }


  /*! set the vertex array (if vector size is 1), or set/enable
    motion blur via multiple time steps, if vector size >= 0 */
  void TrianglesGeom::setVertices(const std::vector<Buffer::SP> &vertexArrays,
                                  /*! the number of vertices in each time step */
                                  size_t count,
                                  size_t stride,
                                  size_t offset)
  {
    vertex.buffers = vertexArrays;
    vertex.count   = count;
    vertex.stride  = stride;
    vertex.offset  = offset;

    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
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
    index.count  = count;
    index.stride = stride;
    index.offset = offset;
    
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.indexPointer = (CUdeviceptr)indices->getPointer(device) + offset;
    }
  }
  
  void TrianglesGeom::setTexCoord(Buffer::SP texCoords,
                                 size_t count,
                                 size_t stride,
                                 size_t offset)
  {
    texCoord.buffer = texCoords;
    texCoord.count  = count;
    texCoord.stride = stride;
    texCoord.offset = offset;
    
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.texCoordPointer = (CUdeviceptr)texCoords->getPointer(device) + offset;
    }
  }

} // ::owl
