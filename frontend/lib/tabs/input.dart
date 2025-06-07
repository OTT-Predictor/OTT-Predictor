import 'package:flutter/material.dart';
import 'package:flutter/widgets.dart';
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
    this.budget,
    this.runtime,
    this.company,
    this.synopsis,
    this.keywords,
  });

  Map<String, dynamic> toJson() {
    return {
      'title': title,
      'genre': genre,
      'language': language,
      'release_year': releaseYear,
      'release_month': releaseMonth,
      'budget': budget,
      'runtime': runtime,
      'company': company,
      'synopsis': synopsis,
      'keywords': keywords,
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
              onChanged: (year, month, date) {},
            ),
            SizedBox(height: 48),
            /*ProductionBudgetInput(
              onChanged: (int? budget) {},
            ),
            SizedBox(height: 48),
            */
            SummaryInput(
              onChanged: (String summary) {
                inputData.synopsis = summary;
              },
            ),
            const SizedBox(height: 48),
            KeywordInput(
              onChanged: (List<String> keywords) {
                inputData.keywords = keywords;
              },
            ),
            const SizedBox(height: 48),
            RuntimeInput(
              onChanged: (int? runtime) {
                inputData.runtime = runtime;
              },
            ),
            const SizedBox(height: 48),
            ProductionCompanySelector(
              onChanged: (
                List<String> companies,
              ) {
                inputData.company = companies;
              },
            ),
            const SizedBox(height: 48),
          ],
        ),
      ),
    );
  }
}
