#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import torch
import numpy as np

from dev_misc import Map

PAD_ID = 0
SOW_ID = 1
EOW_ID = 2
UNK_ID = 3

PAD = '<PAD>'
SOW = '<SOW>'
EOW = '<EOW>'
UNK = '<UNK>'
START_CHAR = [PAD, SOW, EOW, UNK]

_CHARSETS = dict()


def register_charset(lang):
    global _CHARSETS

    def decorated(cls):
        assert lang not in _CHARSETS
        _CHARSETS[lang] = cls
        return cls

    return decorated


def get_charset(lang):
    '''
    Make sure only one charset is ever created.
    '''
    global _CHARSETS
    cls_or_obj = _CHARSETS[lang]
    if isinstance(cls_or_obj, type):
        _CHARSETS[lang] = cls_or_obj()
    return _CHARSETS[lang]


def _recursive_map(func, lst):
    ret = list()
    for item in lst:
        if isinstance(item, (list, np.ndarray)):
            ret.append(_recursive_map(func, item))
        else:
            ret.append(func(item))
    return ret


class BaseCharset(object):

    _CHARS = u''
    _FEATURES = []

    def __init__(self):
        self._id2char = START_CHAR + list(self.__class__._CHARS)
        self._char2id = dict(zip(self._id2char, range(len(self._id2char))))
        self._feat_dict = {}
        for f in self.features:
            self._feat_dict['char'] = None
            self._feat_dict[f] = False

    def __len__(self):
        return len(self._id2char)

    def char2id(self, char):
        def map_func(c): return self._char2id.get(c, UNK_ID)
        if isinstance(char, str):
            return map_func(char)
        elif isinstance(char, (np.ndarray, list)):
            return np.asarray(_recursive_map(map_func, char))
            # return np.asarray([np.asarray(list(map(map_func, ch))) for ch in char])
        else:
            raise NotImplementedError

    def id2char(self, id_):
        def map_func(i): return self._id2char[i]
        if isinstance(id_, int):
            return map_func(id_)
        elif isinstance(id_, (np.ndarray, list)):
            return np.asarray(_recursive_map(map_func, id_))
            # id_.tolist()
            # if id_.ndim == 2:
            #     return np.asarray([np.asarray(list(map(map_func, i))) for i in id_])
            # elif id_.ndim == 3:
            #     return np.asarray([self.id2char(i) for i in id_])
        else:
            raise NotImplementedError

    def get_tokens(self, ids):
        if torch.is_tensor(ids):
            ids = ids.cpu().numpy()
        chars = self.id2char(ids)

        def get_2d_tokens(chars):
            tokens = list()
            for char_seq in chars:
                token = ''
                for c in char_seq:
                    if c == EOW:
                        break
                    elif c in START_CHAR:
                        c = '|'
                    token += c
                tokens.append(token)
            return np.asarray(tokens)

        if chars.ndim == 3:
            a, b, _ = chars.shape
            chars = chars.reshape(a * b, -1)
            tokens = get_2d_tokens(chars).reshape(a, b)
        else:
            tokens = get_2d_tokens(chars)
        return tokens

    def process(self, word):
        # How to process chars in word. This function is language-dependent.
        raise NotImplementedError

    @property
    def features(self):
        return self._FEATURES


@register_charset('en')
class EnCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyz'
    _FEATURES = ['capitalization']

    def process(self, word):
        ret = [copy.deepcopy(Map(self._feat_dict)) for _ in range(len(word))]
        for (i, c) in enumerate(word):
            if c in self._char2id:
                ret[i].update({'char': c})
            else:
                c_lower = c.lower()
                if c_lower in self._char2id:
                    ret[i].update({'char': c_lower})
                    ret[i].update({'capitalization': True})
                else:
                    ret[i].update({'char': ''})
        return ret


@register_charset('es')
class EsCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnñopqrstuvwxyz'
    _FEATURES = ['capitalization']


@register_charset('es-ipa')
class EsIpaCharSet(BaseCharset):

    _CHARS = u'abdefgiklmnoprstuwxúɲɾʎʝʧ'
    _FEATURES = ['']


@register_charset('it')
class ItCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzàèéìïòöù'
    _FEATURES = ['capitalization']


@register_charset('it-ipa')
class ItIpaCharSet(BaseCharset):

    _CHARS = u'abdefghijklmnopqrstuvwzŋɔɛɲʃʎʤʧ'
    _FEATURES = ['']


@register_charset('pt')
class PtCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzáâãçéêíóôú'
    _FEATURES = ['capitalization']


@register_charset('pt-ipa')
class PtIpaCharSet(BaseCharset):

    _CHARS = u'abdefgiklmnopstuvzäɐɔɛɨɾʁʃʎʒ'
    _FEATURES = ['']


@register_charset('heb')
class HebCharSet(BaseCharset):

    _CHARS = u'#$&-<HSTabdghklmnpqrstwyz'
    _FEATURES = ['']


@register_charset('uga')
class UgaCharSet(BaseCharset):

    _CHARS = u'#$&*-<@HSTZabdghiklmnpqrstuvwxyz'
    _FEATURES = ['']


@register_charset('heb-no_spe')
class HebCharSetNoSpe(BaseCharset):

    _CHARS = u'$&<HSTabdghklmnpqrstwyz'
    _FEATURES = ['']


@register_charset('uga-no_spe')                                     
class UgaCharSetNoSpe(BaseCharset):

    _CHARS = u'$&*<@HSTZabdghiklmnpqrstuvwxyz'
    _FEATURES = ['']


#@register_charset('kor')                                                            # hs 20240129
#class KorCharSetNoSpe(BaseCharset):

#    _CHARS = u'가각간갈감갑값갓강개객갱갹거건걸검겁겉게격견결겸경곁계고곡곤골곰곳공과곽관괄광괘괴괵굉교구국군굴굿궁권궐궤귀규균귤극근글금급긍기긱긴길김깃껏꼴꽃끈끌나낙난날남납낭내냥널네녀년녈념녕노논놈농뇌뇨눈눌뉴뉵능늪니닉님다단달담답당대댁덕덩도독돈돌동되두둑둔득들등딸때떡떼뜀뜰띠라락란랄람랍랑래랭략량려력련렬렴렵령례로록론롱뢰료룡루류륙륜률륭르륵름릉리린림립마막만말맛망매맥맹먹메멱면멸명모목몰못몽묘무묵문물미민밀밑바박반발밤밥방밭배백번벌범법벽변별병보복본봉부북분불붓붕비빈빗빙빛뼈뿔사삭산살삼삽상새색생서석선설섬섭성섶세셈소속손솔솜송솥쇄쇠수숙순술숭숲슬습승시식신실심십쌀쌈쌍쑤씨아악안알암압앙애액앵야약양어억언얼엄업에여역연열염엽영예오옥온올옷옹와완왈왕왜외요욕용우욱운울웅원월위유육윤융은을음읍응의이익인일임입잉잎자작잔잠잡장재쟁저적전절점접젓정젖제조족존졸좀종좌좨죄주죽준줄중쥐즉즐즙증지직진질짐집짓징짜짝쪽찜차착찬찰참창채책처척천철첨첩청체초촉촌총촬최추축춘출춤충췌취측츤츰층치칙친칠침칩칭칸칼코콩쾌타탁탄탈탐탑탕태택탱터턱털테토통퇴투특튼틀틈파판팔패팽퍅페편폄평포폭표푼풀품풍피필핍하학한할함합항해핵행향허헌헐험혀혁현혈혐협형혜호혹혼홀홈홍화확환활황회획횡효후훈훙훤훼휘휴휼흉흑흔흘흙흠흡흥희히힐힘'
#    _FEATURES = ['']


#@register_charset('wuu')                                                            # hs 20240129
#class WuuCharSetNoSpe(BaseCharset):

#    _CHARS = u'abdefghijklmnopqrstuvxyz'
#    _FEATURES = ['']

@register_charset('kor')                                                            # hs 20240307 IPA for kor
class KorCharSetNoSpe(BaseCharset):

    _CHARS = u'*abdhijklmnopstuwŋɑɕɛɡɯɰɾʌʑʰ'
    _FEATURES = ['']


@register_charset('wuu')                                                            # hs 20240307 IPA for wuu
class WuuCharSetNoSpe(BaseCharset):

    _CHARS = u'abdefhiklmnopstvyzãøŋɑɔɕəɜɡɥɦɪɯɲɻʊʏʑʔʰʲʷ̞̠̥̩̯̱̃̊̍͡'
    _FEATURES = ['']

@register_charset('OC')                                                             # hs 20240307
class OCCharSet(BaseCharset):

    _CHARS = u'ACNSabdeghijklmnopqrstuwz|ŋəɢɦʔʰʷˤr̥l̥n̥'                               # hs 20240310
    _FEATURES = ['']

@register_charset('MC')                                                             # hs 20240307
class MCCharSet(BaseCharset):

    _CHARS = u'+abdeghijklmnoprstuwxyz'
    _FEATURES = ['']

@register_charset('el')
class ElCharSet(BaseCharset):

    _CHARS = u'fhyαβγδεζηθικλμνξοπρςστυφχψω'
    _FEATURES = ['']


@register_charset('linb-latin')
class LinbLatinCharSet(BaseCharset):

    _CHARS = u'23adeijkmnopqrstuwz'
    _FEATURES = ['']


@register_charset('minoan')
class MinoanCharSet(BaseCharset):

    _CHARS = u'𐀀𐀁𐀂𐀃𐀄𐀅𐀆𐀇𐀈𐀉𐀊𐀋𐀍𐀏𐀐𐀑𐀒𐀓𐀔𐀕𐀖𐀗𐀘𐀙𐀚𐀛𐀜𐀝𐀞𐀟𐀠𐀡𐀢𐀣𐀤𐀥𐀦𐀨𐀩𐀪𐀫𐀬𐀭𐀮𐀯𐀰𐀱𐀲𐀳𐀴𐀵𐀶𐀷𐀸𐀹𐀺𐀼𐀽𐀿𐁀𐁁𐁂𐁄𐁅𐁆𐁇𐁈𐁉𐁊𐁋'
    _FEATURES = ['']


@register_charset('fr')
class FrCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyz'
    _FEATURES = ['capitalization']


@register_charset('lost')
class LostCharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('k1')
class K1CharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('k2')
class K2CharSet(BaseCharset):
    _CHARS = u'aeioubpdtfvgklmnrszw'


@register_charset('de')
class DeCharSet(BaseCharset):

    _CHARS = u'abcdefghijklmnopqrstuvwxyzäöüß'
    _FEATURES = ['capitalization', 'umlaut']
    _UMLAUT = (u'ä', u'ö', u'ü')

    def process(self, word):
        ret = [copy.deepcopy(Map(self._feat_dict)) for _ in range(len(word))]
        for (i, c) in enumerate(word):
            if c in self._char2id:
                ret[i].update({'char': c})
                if c in self.__class__._UMLAUT:
                    ret[i].update({'umlaut': True})
            else:
                c_lower = c.lower()
                if c_lower in self.__class__._UMLAUT:
                    ret[i].update({'umlaut': True})
                if c_lower in self._char2id:
                    ret[i].update({'char': c_lower})
                    ret[i].update({'capitalization': True})
                else:
                    ret[i].update({'char': ''})
        return ret
