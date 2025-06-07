import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';
import 'package:ossw4_msps/tabs/gerne.dart';

class RankingTab extends StatelessWidget {
  const RankingTab({super.key});

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
        child: Column(
          crossAxisAlignment:
              CrossAxisAlignment.start,
          children: [SizedBox(height: 64)],
        ),
      ),
    );
  }
}
