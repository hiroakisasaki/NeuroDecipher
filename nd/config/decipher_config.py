from typing import Tuple

from . import registry

register = registry.register


@register
class UgaHebSmallNoSpe:
    lost_lang: str = 'uga-no_spe'
    known_lang: str = 'heb-no_spe'
    #cog_path: str = '/home/s2230007/venv/python3.7/root/NeuroDecipher/data/uga-heb.small.no_spe.cog'   # hs 20221122 kagayaki
    cog_path: str = 'data/uga-heb.small.no_spe.cog'                                                     # hs 20240109 colab
    num_cognates: int = 221
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5
    #gpu: int = 0                                                                                        # hs 20240109 #kagayaki

@register
class KorWuu:                                                                                           # hs 20240129
    lost_lang: str = 'kor'
    known_lang: str = 'wuu'
    #cog_path: str = '/home/s2230007/venv/python3.7/root/NeuroDecipher/data/cognate.txt'                # hs 20240129 kagayaki
    cog_path: str = 'data/cognate.txt'                                                                  # hs 20240129 colab
    num_cognates: int = 220
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5
    #gpu: int = 0                                                                                       # hs 20240207 kagayaki            

@register
class OCMC:                                                                                             # hs 20240307
    lost_lang: str = 'OC'
    known_lang: str = 'MC'
    #cog_path: str = '/home/s2230007/venv/python3.7/root/NeuroDecipher/data/OCMC.txt'                   # hs 20240307 kagayaki
    cog_path: str = 'data/OCMC.txt'                                                                     # hs 20240307 colab
    num_cognates: int = 4967
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5
    #gpu: int = 0                                                                                       # hs 20240307 kagayaki            

@register
class OCMCtri:                                                                                          # hs 20240314
    lost_lang: str = 'OC'
    known_lang: str = 'MC'
    #cog_path: str = '/home/s2230007/venv/python3.7/root/NeuroDecipher/data/OCMCtri.txt'                # hs 20240314 kagayaki
    cog_path: str = 'data/OCMCtri.txt'                                                                  # hs 20240314 colab
    num_cognates: int = 221
    num_epochs_per_M_step: int = 150
    eval_interval: int = 10
    check_interval: int = 10
    num_rounds: int = 10
    batch_size: int = 500
    n_similar: int = 5
    capacity: int = 3
    dropout: float = 0.3
    warm_up_steps: int = 5
    #gpu: int = 0                                                                                       # hs 20240314 kagayaki            
