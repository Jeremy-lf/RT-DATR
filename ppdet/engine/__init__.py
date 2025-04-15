# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from . import trainer
from .trainer import *

from . import trainer_cot
from .trainer_cot import *

from . import callbacks
from .callbacks import *

from . import env
from .env import *

__all__ = trainer.__all__ \
        + callbacks.__all__ \
        + env.__all__

from . import tracker
from .tracker import *
__all__ = __all__ + tracker.__all__

from . import trainer_ssod
from .trainer_ssod import *
__all__ = __all__ + trainer_ssod.__all__

from . import trainer_da
from .trainer_da import *
__all__ = __all__ + trainer_da.__all__

from . import trainer_da_dynamic
from .trainer_da_dynamic import *
__all__ = __all__ + trainer_da_dynamic.__all__


from . import trainer_da_weak
from .trainer_da_weak import *
__all__ = __all__ + trainer_da_weak.__all__

from . import trainer_da_instance
from .trainer_da_instance import *
__all__ = __all__ + trainer_da_instance.__all__

from . import trainer_da_align
from .trainer_da_align import *
__all__ = __all__ + trainer_da_align.__all__


from . import trainer_da_backbone
from .trainer_da_backbone import *
__all__ = __all__ + trainer_da_backbone.__all__


from . import trainer_da_backbone_encoder
from .trainer_da_backbone_encoder import *
__all__ = __all__ + trainer_da_backbone_encoder.__all__


from . import trainer_da_backbone_encoder_instance
from .trainer_da_backbone_encoder_instance import *
__all__ = __all__ + trainer_da_backbone_encoder_instance.__all__

from . import trainer_da_backbone_encoder_instance_dn
from .trainer_da_backbone_encoder_instance_dn import *
__all__ = __all__ + trainer_da_backbone_encoder_instance_dn.__all__


from . import trainer_da_backbone_encoder_instance_dn_cmt
from .trainer_da_backbone_encoder_instance_dn_cmt import *
__all__ = __all__ + trainer_da_backbone_encoder_instance_dn_cmt.__all__