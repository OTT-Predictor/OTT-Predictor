import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class GenreSelector extends StatefulWidget {
  const GenreSelector({super.key});

  @override
  State<GenreSelector> createState() =>
      _GenreSelectorState();
}

class _GenreSelectorState
    extends State<GenreSelector> {
  final List<String> genres = [
    "SF",
    "TV_영화",
    "가족",
    "공포",
    "다큐멘터리",
    "드라마",
    "로맨스",
    "모험",
    "미스터리",
    "범죄",
    "서부",
    "스릴러",
    "애니메이션",
    "액션",
    "역사",
    "음악",
    "전쟁",
    "코미디",
    "판타지",
  ];

  Set<String> selectedGenres = {};

  void toggleGenre(String genre) {
    setState(() {
      if (selectedGenres.contains(genre)) {
        selectedGenres.remove(genre);
      } else {
        selectedGenres.add(genre);
      }
    });
  }

  void clearGenres() {
    setState(() {
      selectedGenres.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text(
              "장르(다중선택)",
              style: subtitleText,
            ),
            Spacer(),
            ElevatedButton(
              onPressed: clearGenres,
              child: Text(
                '초기화',
                style: TextStyle(
                  fontFamily: 'Pretendard',
                  fontSize: 16,
                  color: Color(0xFF333333),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 16),

        Wrap(
          spacing: 12,
          runSpacing: 12,
          children:
              genres.map((genre) {
                final isSelected = selectedGenres
                    .contains(genre);
                return GestureDetector(
                  onTap: () => toggleGenre(genre),
                  child: Container(
                    padding:
                        const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 12,
                        ),
                    decoration: BoxDecoration(
                      color:
                          isSelected
                              ? Colors.purple[200]
                              : Colors.white,
                      border: Border.all(
                        color: Colors.black12,
                      ),
                      borderRadius:
                          BorderRadius.circular(
                            12,
                          ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black12,
                          blurRadius: 4,
                          offset: Offset(0, 2),
                        ),
                      ],
                    ),
                    child: Text(
                      genre,
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight:
                            isSelected
                                ? FontWeight.bold
                                : FontWeight
                                    .normal,
                        color:
                            isSelected
                                ? Colors.white
                                : Colors.black87,
                      ),
                    ),
                  ),
                );
              }).toList(),
        ),

        const SizedBox(height: 24),
      ],
    );
  }
}
