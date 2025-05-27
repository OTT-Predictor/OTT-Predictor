import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class ReleaseMonthSelector
    extends StatefulWidget {
  final Function(int?, int?) onChanged;

  const ReleaseMonthSelector({
    super.key,
    required this.onChanged,
  });

  @override
  State<ReleaseMonthSelector> createState() =>
      _ReleaseMonthSelectorState();
}

class _ReleaseMonthSelectorState
    extends State<ReleaseMonthSelector> {
  int? selectedYear;
  int? selectedMonth;

  List<int> get years {
    //final currentYear = DateTime.now().year;
    return List.generate(
      51,
      (index) => 2040 - index,
    );
  }

  final List<int> months = List.generate(
    12,
    (index) => 12 - index,
  );

  void _onSelectionChanged() {
    widget.onChanged(selectedYear, selectedMonth);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("개봉 연월", style: subtitleText),
        const SizedBox(height: 8),
        Row(
          children: [
            // 년도 드롭다운
            DropdownButton<int>(
              hint: const Text("년도"),
              value: selectedYear,
              items:
                  years.map((year) {
                    return DropdownMenuItem<int>(
                      value: year,
                      child: Text("$year년"),
                    );
                  }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedYear = value;
                  _onSelectionChanged();
                });
              },
            ),
            const SizedBox(width: 16),
            // 월 드롭다운
            DropdownButton<int>(
              hint: const Text("월"),
              value: selectedMonth,
              items:
                  months.map((month) {
                    return DropdownMenuItem<int>(
                      value: month,
                      child: Text("$month월"),
                    );
                  }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedMonth = value;
                  _onSelectionChanged();
                });
              },
            ),
          ],
        ),
      ],
    );
  }
}
