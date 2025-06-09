import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:flutter/services.dart'
    show rootBundle;
import 'package:ossw4_msps/main.dart';

class RankingPage extends StatefulWidget {
  const RankingPage({super.key});

  @override
  _RankingPageState createState() =>
      _RankingPageState();
}

class _RankingPageState
    extends State<RankingPage> {
  List<Map<String, dynamic>> rankingData = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    loadRankingData();
  }

  Future<void> loadRankingData() async {
    try {
      final rawData = await rootBundle.loadString(
        'assets/all_movies_predictions.csv',
      );
      final lines = LineSplitter().convert(
        rawData.trim(),
      );

      final rows =
          lines
              .skip(1)
              .map((line) {
                final values = line.split(',');
                if (values.length < 2)
                  return null;
                return {
                  'title': values[0],
                  'prob':
                      double.tryParse(
                        values[1].trim(),
                      ) ??
                      0.0,
                };
              })
              .whereType<Map<String, dynamic>>()
              .toList();

      rows.sort(
        (a, b) => (b['prob'] as double).compareTo(
          a['prob'] as double,
        ),
      );

      setState(() {
        rankingData = rows.take(100).toList();
        _isLoading = false;
      });
    } catch (e) {
      print('랭킹 데이터 로드 오류: $e');
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
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
        child:
            _isLoading
                ? Padding(
                  padding: const EdgeInsets.all(
                    32.0,
                  ),
                  child: Center(
                    child:
                        CircularProgressIndicator(),
                  ),
                )
                : ListView.separated(
                  shrinkWrap: true,
                  physics:
                      NeverScrollableScrollPhysics(),
                  itemCount: rankingData.length,
                  separatorBuilder:
                      (context, index) =>
                          Divider(height: 1),
                  itemBuilder: (context, index) {
                    final movie =
                        rankingData[index];
                    return Padding(
                      padding:
                          const EdgeInsets.symmetric(
                            vertical: 12.0,
                          ),
                      child: Row(
                        mainAxisAlignment:
                            MainAxisAlignment
                                .spaceBetween,
                        children: [
                          Text(
                            '#${index + 1}',
                            style: TextStyle(
                              fontWeight:
                                  FontWeight.bold,
                              fontSize: 18,
                            ),
                          ),
                          Expanded(
                            child: Padding(
                              padding:
                                  const EdgeInsets.symmetric(
                                    horizontal:
                                        16,
                                  ),
                              child: Text(
                                movie['title'],

                                overflow:
                                    TextOverflow
                                        .ellipsis,
                              ),
                            ),
                          ),
                          Text(
                            '${(movie['prob'] * 100).toStringAsFixed(2)}%',
                            style: TextStyle(
                              color:
                                  Colors
                                      .blueAccent,
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),
      ),
    );
  }
}
