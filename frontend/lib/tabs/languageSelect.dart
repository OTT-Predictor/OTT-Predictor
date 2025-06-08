import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class LanguageSelector extends StatefulWidget {
  final Function(String?) onChanged;
  const LanguageSelector({
    super.key,
    required this.onChanged,
  });

  @override
  State<LanguageSelector> createState() =>
      _LanguageSelectorState();
}

class _LanguageSelectorState
    extends State<LanguageSelector> {
  final Map<String, String> languageMap = {
    'gl': '갈리시아어',
    'gu': '구자라트어',
    'el': '그리스어',
    'nl': '네덜란드어',
    'ne': '네팔어',
    'no': '노르웨이어',
    'nb': '노르웨이어(보크말)',
    'da': '덴마크어',
    'de': '독일어',
    'dv': '디베히어',
    'lv': '라트비아어',
    'la': '라틴어',
    'ru': '러시아어',
    'ro': '루마니아어',
    'lt': '리투아니아어',
    'mr': '마라티어',
    'mi': '마오리어',
    'mk': '마케도니아어',
    'ml': '말라얄람어',
    'ms': '말레이어',
    'mn': '몽골어',
    'eu': '바스크어',
    'bm': '밤바라어',
    'vi': '베트남어',
    'be': '벨라루스어',
    'bn': '벵골어',
    'bs': '보스니아어',
    'bg': '불가리아어',
    'sr': '세르비아어',
    'sw': '스와힐리어',
    'sv': '스웨덴어',
    'es': '스페인어',
    'sk': '슬로바키아어',
    'sl': '슬로베니아어',
    'si': '신할라어',
    'ar': '아랍어',
    'hy': '아르메니아어',
    'is': '아이슬란드어',
    'ga': '아일랜드어',
    'az': '아제르바이잔어',
    'af': '아프리칸스어',
    'am': '암하라어',
    'et': '에스토니아어',
    'en': '영어',
    'ur': '우르두어',
    'uk': '우크라이나어',
    'cy': '웨일스어',
    'iu': '이누이트어(이누크티투트)',
    'it': '이탈리아어',
    'id': '인도네시아어',
    'ja': '일본어',
    'ka': '조지아어',
    'dz': '종카어',
    'zu': '줄루어',
    'zh': '중국어',
    'cn': '중국어(비표준)',
    'cs': '체코어',
    'tn': '츠와나어',
    'ks': '카슈미리어',
    'kk': '카자흐어',
    'ca': '카탈로니아어',
    'kn': '칸나다어',
    'xh': '코사어',
    'ku': '쿠르드어',
    'hr': '크로아티아어',
    'km': '크메르어',
    'tl': '타갈로그어(필리핀어)',
    'ta': '타밀어',
    'tg': '타지크어',
    'th': '태국어',
    'tr': '터키어',
    'te': '텔루구어',
    'ps': '파슈토어',
    'pa': '펀자브어',
    'fa': '페르시아어',
    'pt': '포르투갈어',
    'pl': '폴란드어',
    'ff': '풀라어',
    'fr': '프랑스어',
    'fi': '핀란드어',
    'ko': '한국어',
    'hu': '헝가리어',
    'he': '히브리어',
    'hi': '힌디어',
  };

  String? selectedCode;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("언어", style: subtitleText),
        const SizedBox(height: 8),
        DropdownButton<String>(
          isExpanded: true,
          value: selectedCode,
          hint: const Text("언어를 선택하세요"),
          items:
              languageMap.entries.map((entry) {
                return DropdownMenuItem<String>(
                  value: entry.key,
                  child: Text(entry.value),
                );
              }).toList(),
          onChanged: (value) {
            setState(() {
              selectedCode = value;
              widget.onChanged(value); // 외부로 전달
            });
          },
        ),
      ],
    );
  }
}
