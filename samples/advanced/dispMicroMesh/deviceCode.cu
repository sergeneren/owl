// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "deviceCode.h"
#include <optix_device.h>
#include <optix_micromap.h>

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

  vec3f color;
  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/color);
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData& self = owl::getProgramData<TrianglesGeomData>();

  float3 vertices[3];
  vec3f hitP;
  vec3f Ng;

  if (optixIsTriangleHit())
  {
      optixGetTriangleVertexData(optixGetGASTraversableHandle(), optixGetPrimitiveIndex(), optixGetSbtGASIndex(),
          optixGetRayTime(), vertices);

      float2 barycentrics = optixGetTriangleBarycentrics();

      vec3f vertex0 = vertices[0];
      vec3f vertex1 = vertices[1];
      vec3f vertex2 = vertices[2];

      Ng = normalize(cross(vertex1 - vertex0, vertex2 - vertex0));
      hitP = (1.0f - barycentrics.x - barycentrics.y) * vertex0 + barycentrics.x * vertex1 + barycentrics.y * vertex2;
  }
  else if (optixIsDisplacedMicromeshTriangleHit())
  {
      // returns the vertices of the current DMM micro triangle hit
      optixGetMicroTriangleVertexData(vertices);

      float2 hitBaseBarycentrics = optixGetTriangleBarycentrics();

      float2 microVertexBaseBarycentrics[3];
      optixGetMicroTriangleBarycentricsData(microVertexBaseBarycentrics);

      float2 microBarycentrics = optixBaseBarycentricsToMicroBarycentrics(hitBaseBarycentrics, microVertexBaseBarycentrics);

      vec3f vertex0 = vertices[0];
      vec3f vertex1 = vertices[1];
      vec3f vertex2 = vertices[2];

      Ng = normalize(cross(vertex1 - vertex0, vertex2 - vertex0));
      hitP = (1.0f - microBarycentrics.x - microBarycentrics.y) * vertex0 + microBarycentrics.x * vertex1 + microBarycentrics.y * vertex2;
  }

  prd = normalize(hitP);
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}

