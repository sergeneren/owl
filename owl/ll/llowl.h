// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

#pragma once

/*! \file owl/ll/owl-ll.h Main "api" functoin that swtiches between
    host- and device-side apis based on whether we're compiling for
    device- or host-side */

#include <optix.h>
#include <gdt/math/vec.h>
#include <gdt/math/box.h>

namespace owl {
  using namespace gdt;
}

#ifdef __CUDACC__
#  include "deviceAPI.h"
#else
#  include "DeviceGroup.h"
#endif
