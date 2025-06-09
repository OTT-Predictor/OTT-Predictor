import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
import 'package:http/http.dart' as http;

import 'package:ossw4_msps/main.dart';
import 'package:ossw4_msps/tabs/gerne.dart';
import 'package:ossw4_msps/tabs/languageSelect.dart';
import 'package:ossw4_msps/tabs/dateSelect.dart';
import 'package:ossw4_msps/tabs/expense.dart';
import 'package:ossw4_msps/tabs/summary.dart';
import 'package:ossw4_msps/tabs/tag.dart';
import 'package:ossw4_msps/tabs/runtime.dart';
import 'package:ossw4_msps/tabs/company.dart';

class MovieInput {
  String? title;
  List<String>? genre;
  String? language;
  int? releaseYear;
  int? releaseMonth;
  int? releaseDay;
  int? budget;
  int? runtime;
  List<String>? company;
  String? synopsis;
  List<String>? keywords;

  MovieInput({
    this.title,
    this.genre,
    this.language,
    this.releaseYear,
    this.releaseMonth,
    this.releaseDay,
    this.budget,
    this.runtime,
    this.company,
    this.synopsis,
    this.keywords,
  });

  Map<String, dynamic> toJson() {
    return {
      'title': title,
      'synopsis': synopsis,
      'keywords': keywords,
      'runtime': runtime,
      'release_year': releaseYear,
      'release_month': releaseMonth,
      'release_day': releaseDay,
      'language': language,
      'company': company,
      'genre': genre,
    };
  }
}

class InputTab extends StatefulWidget {
  const InputTab({super.key});

  @override
  State<InputTab> createState() =>
      _InputTabState();
}

class _InputTabState extends State<InputTab> {
  final MovieInput inputData = MovieInput();
  String? selectedLanguageCode;

  void updateSelectedLanguage(String? code) {
    setState(() {
      selectedLanguageCode = code;
      inputData.language = code;
    });
  }

  @override
  Widget build(BuildContext context) {
    final TextEditingController inputCon =
        TextEditingController();
    double pageWidth =
        MediaQuery.of(context).size.width;
    double horizontalPadding =
        pageWidth > breakPointWidth
            ? (pageWidth - breakPointWidth) / 2
            : 20;

    return Padding(
      padding: EdgeInsets.symmetric(
        horizontal: horizontalPadding,
      ),
      child: Container(
        padding: EdgeInsets.symmetric(
          horizontal: 48,
        ),
        color: Colors.white,
        width: double.infinity,
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.start,
            children: [
              SizedBox(height: 48),
              Text("입력", style: titleText),
              SizedBox(height: 32),
              Text("제목", style: subtitleText),
              SizedBox(height: 16),
              TextField(
                controller: inputCon,
                decoration: InputDecoration(
                  suffixIcon: IconButton(
                    icon: Icon(Icons.close),
                    onPressed: () {
                      inputCon.clear();
                      setState(() {});
                    },
                  ),
                  labelText: '제목',
                  labelStyle: inputText,
                  border: OutlineInputBorder(),
                ),
                onChanged: (value) {
                  inputData.title = value;
                },
              ),
              SizedBox(height: 48),
              GenreSelector(
                onChanged: (List<String> genres) {
                  inputData.genre = genres;
                },
              ),
              SizedBox(height: 48),
              LanguageSelector(
                onChanged: updateSelectedLanguage,
              ),
              SizedBox(height: 48),
              ReleaseDateSelector(
                onChanged: (year, month, day) {
                  inputData.releaseYear = year;
                  inputData.releaseMonth = month;
                  inputData.releaseDay = day;
                },
              ),
              SizedBox(height: 48),
              SummaryInput(
                onChanged: (String summary) {
                  inputData.synopsis = summary;
                },
              ),
              SizedBox(height: 48),
              KeywordInput(
                onChanged: (
                  List<String> keywords,
                ) {
                  inputData.keywords = keywords;
                },
              ),
              SizedBox(height: 48),
              RuntimeInput(
                onChanged: (int? runtime) {
                  inputData.runtime = runtime;
                },
              ),
              SizedBox(height: 48),
              ProductionCompanySelector(
                onChanged: (
                  List<String> companies,
                ) {
                  inputData.company = companies;
                },
              ),
              SizedBox(height: 48),
              ElevatedButton(
                onPressed: () async {
                  if (inputData.title == null ||
                      inputData.title!.isEmpty ||
                      inputData.synopsis ==
                          null ||
                      inputData
                          .synopsis!
                          .isEmpty ||
                      inputData.keywords ==
                          null ||
                      inputData
                          .keywords!
                          .isEmpty ||
                      inputData.runtime == null ||
                      inputData.releaseYear ==
                          null ||
                      inputData.releaseMonth ==
                          null ||
                      inputData.releaseDay ==
                          null ||
                      inputData.language ==
                          null ||
                      inputData
                          .language!
                          .isEmpty ||
                      inputData.genre == null ||
                      inputData.genre!.isEmpty) {
                    showDialog(
                      context: context,
                      builder:
                          (_) => AlertDialog(
                            title: Text('오류'),
                            content: Text(
                              '내용을 더 입력해주세요.',
                            ),
                            actions: [
                              TextButton(
                                onPressed:
                                    () =>
                                        Navigator.pop(
                                          context,
                                        ),
                                child: Text('확인'),
                              ),
                            ],
                          ),
                    );
                    return;
                  }

                  final requestBody = {
                    "title": inputData.title,
                    "synopsis":
                        inputData.synopsis,
                    "keywords":
                        inputData.keywords,
                    "runtime": inputData.runtime,
                    "release_year":
                        inputData.releaseYear,
                    "release_month":
                        inputData.releaseMonth,
                    "release_day":
                        inputData.releaseDay,
                    "language":
                        inputData.language,
                    "company":
                        inputData.company ?? [],
                    "genre": inputData.genre,
                  };

                  try {
                    final response = await http.post(
                      Uri.parse(
                        'http://127.0.0.1:8000/predict',
                      ),
                      headers: {
                        'Content-Type':
                            'application/json',
                      },
                      body: jsonEncode(
                        requestBody,
                      ),
                    );

                    if (response.statusCode ==
                        200) {
                      final decoded = utf8.decode(
                        response.bodyBytes,
                      ); // ← 여기!
                      final responseData =
                          jsonDecode(decoded);
                      final probability =
                          responseData['success_probability'];
                      final verdict =
                          responseData['verdict'];

                      showDialog(
                        context: context,
                        builder:
                            (_) => AlertDialog(
                              title: Text(
                                '예측 결과',
                              ),
                              content: Text(
                                '성공 확률: ${(probability * 100).toStringAsFixed(2)}%\n판단: $verdict',
                              ),
                              actions: [
                                TextButton(
                                  onPressed:
                                      () => Navigator.pop(
                                        context,
                                      ),
                                  child: Text(
                                    '닫기',
                                  ),
                                ),
                              ],
                            ),
                      );
                    } else {
                      showDialog(
                        context: context,
                        builder:
                            (_) => AlertDialog(
                              title: Text('에러'),
                              content: Text(
                                '서버 오류: ${response.statusCode}',
                              ),
                              actions: [
                                TextButton(
                                  onPressed:
                                      () => Navigator.pop(
                                        context,
                                      ),
                                  child: Text(
                                    '닫기',
                                  ),
                                ),
                              ],
                            ),
                      );
                    }
                  } catch (e) {
                    // 예외 발생 시
                    showDialog(
                      context: context,
                      builder:
                          (_) => AlertDialog(
                            title: Text('요청 실패'),
                            content: Text(
                              '서버에 연결할 수 없습니다.\n에러: $e',
                            ),
                            actions: [
                              TextButton(
                                onPressed:
                                    () =>
                                        Navigator.pop(
                                          context,
                                        ),
                                child: Text('확인'),
                              ),
                            ],
                          ),
                    );
                  }
                },
                child: Text('예측 요청'),
              ),
              SizedBox(height: 48),
            ],
          ),
        ),
      ),
    );
  }
}
