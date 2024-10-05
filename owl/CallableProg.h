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

#include "SBTObject.h"
#include "Module.h"

namespace owl {

  /*! type that describes a callable program's variables and programs */
  struct CallableProgType : public SBTObjectType {
    typedef std::shared_ptr<CallableProgType> SP;
    CallableProgType(Context *const context,
               Module::SP module,
               const std::string &dcName,
               const std::string &ccName,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    /*! for callable progs there's exactly one programgroup pre object */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;
      
      /*! constructor, only pass-through to parent class */
      DeviceData(const DeviceContext::SP &device);

      /*! the optix-compiled program group witin the given device's
        optix context */
      OptixProgramGroup pg = 0;
    };

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
    
    /*! the module in which the program is defined */
    Module::SP module;

    /*! the name, annotated with optix' "__direct_callable__" */
    const std::string DirectCallableName;

    /*! the name, annotated wihth optix' "__continuation_callable__" */
    const std::string ContinuationCallableName;
  };
  
  /*! an actual instance of a callable program - defined by its type and
      variable values */
  struct CallableProg : public SBTObject<CallableProgType> {
    typedef std::shared_ptr<CallableProg> SP;

    /*! constructor */
    CallableProg(Context *const context,
           CallableProgType::SP type);
    
    /*! write the given SBT record, using the given device's
      corresponding device-side data represenataion */
    void writeSBTRecord(uint8_t *const sbtRecord,
                        const DeviceContext::SP &device);
    
    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
  };
  
  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline CallableProgType::DeviceData &CallableProgType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }
} // ::owl

