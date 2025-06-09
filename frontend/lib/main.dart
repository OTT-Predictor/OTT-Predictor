import 'package:flutter/material.dart';
import 'tabs/input.dart';
import 'tabs/ranking.dart';

const TextStyle titleText = TextStyle(
  color: Color(0xFF111111),
  fontFamily: 'Pretendard',
  fontSize: 48,
  fontWeight: FontWeight.w600,
);
const TextStyle selectText = TextStyle(
  color: Color(0xFF111111),
  fontFamily: 'Pretendard',
  fontSize: 20,
  fontWeight: FontWeight.w600,
);
const TextStyle subtitleText = TextStyle(
  color: Color(0xFF111111),
  fontFamily: 'Pretendard',
  fontSize: 32,
  fontWeight: FontWeight.w500,
);
const TextStyle inputText = TextStyle(
  color: Color(0xFF111111),
  fontFamily: 'Pretendard',
  fontSize: 20,
  fontWeight: FontWeight.w300,
);
const TextStyle regularText = TextStyle(
  color: Color(0xFF111111),
  fontFamily: 'Pretendard',
  fontSize: 36,
  fontWeight: FontWeight.w400,
);
Color backColor = Color(0xFFF9EFFF);
Color selectedColor = Color(0xFFE1C2F2);
const double breakPointWidth = 1080;
final TextEditingController inputCon =
    TextEditingController();

void main() {
  runApp(const OTTS());
}

class OTTS extends StatelessWidget {
  const OTTS({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(home: mainpage());
  }
}

class mainpage extends StatefulWidget {
  const mainpage({super.key});

  @override
  State<mainpage> createState() =>
      _mainpageState();
}

class _mainpageState extends State<mainpage> {
  bool isNotSelectedTabisRanking = true;

  @override
  Widget build(BuildContext context) {
    double pageWidth =
        MediaQuery.of(context).size.width;
    return Scaffold(
      backgroundColor: backColor,
      body: Column(
        children: [
          Container(
            color: Colors.white,
            width: double.infinity,
            height: 64,
            padding: EdgeInsets.symmetric(
              horizontal:
                  pageWidth > breakPointWidth
                      ? (pageWidth -
                              breakPointWidth) /
                          2
                      : 20,
            ),
            child: Row(
              mainAxisAlignment:
                  MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'OTTS',
                  style: TextStyle(
                    fontFamily: 'Pretendard',
                    fontSize: 36,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                Spacer(),
                Text(
                  '공개SW프로젝트 4조',
                  style: TextStyle(
                    fontFamily: 'Pretendard',
                    fontSize: 24,
                    fontWeight: FontWeight.w400,
                  ),
                ),
              ],
            ),
          ),
          Expanded(
            child: SingleChildScrollView(
              child: Column(
                children: [
                  SizedBox(height: 100),

                  Container(
                    height: 60,
                    padding: EdgeInsets.symmetric(
                      horizontal:
                          pageWidth >
                                  breakPointWidth
                              ? (pageWidth -
                                      breakPointWidth) /
                                  2
                              : 20,
                    ),

                    child: Row(
                      children: [
                        Expanded(
                          child: GestureDetector(
                            onTap:
                                () => setState(
                                  () =>
                                      isNotSelectedTabisRanking =
                                          false,
                                ),
                            child: Container(
                              padding:
                                  const EdgeInsets.all(
                                    16,
                                  ),
                              decoration: BoxDecoration(
                                color:
                                    isNotSelectedTabisRanking ==
                                            false
                                        ? Colors
                                            .white
                                        : Colors
                                            .purple[200],
                                borderRadius:
                                    const BorderRadius.only(
                                      topLeft:
                                          Radius.circular(
                                            16,
                                          ),
                                    ),
                              ),
                              child: const Center(
                                child: Text(
                                  '랭킹',
                                  style:
                                      selectText,
                                ),
                              ),
                            ),
                          ),
                        ),
                        Expanded(
                          child: GestureDetector(
                            onTap:
                                () => setState(
                                  () =>
                                      isNotSelectedTabisRanking =
                                          true,
                                ),
                            child: Container(
                              padding:
                                  const EdgeInsets.all(
                                    16,
                                  ),
                              decoration: BoxDecoration(
                                color:
                                    isNotSelectedTabisRanking ==
                                            true
                                        ? Colors
                                            .white
                                        : Colors
                                            .purple[200],
                                borderRadius:
                                    const BorderRadius.only(
                                      topRight:
                                          Radius.circular(
                                            16,
                                          ),
                                    ),
                              ),
                              child: const Center(
                                child: Text(
                                  '성공 확률 예측',
                                  style:
                                      selectText,
                                ),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                  AnimatedSwitcher(
                    duration: const Duration(
                      milliseconds: 300,
                    ),
                    child:
                        isNotSelectedTabisRanking
                            ? const InputTab()
                            : RankingPage(),
                  ),
                  SizedBox(height: 50),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
