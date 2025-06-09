import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class ReleaseDateSelector extends StatefulWidget {
  final Function(int?, int?, int?) onChanged;

  const ReleaseDateSelector({
    super.key,
    required this.onChanged,
  });

  @override
  State<ReleaseDateSelector> createState() =>
      _ReleaseDateSelectorState();
}

class _ReleaseDateSelectorState
    extends State<ReleaseDateSelector> {
  int? selectedYear;
  int? selectedMonth;
  int? selectedDay;

  List<int> get years {
    final currentYear = DateTime.now().year;
    return List.generate(
      51,
      (index) => currentYear + 51 - index,
    );
  }

  final List<int> months = List.generate(
    12,
    (index) => index + 1,
  );

  List<int> get days {
    if (selectedYear != null &&
        selectedMonth != null) {
      final lastDay =
          DateTime(
            selectedYear!,
            selectedMonth! + 1,
            0,
          ).day;
      return List.generate(lastDay, (i) => i + 1);
    } else {
      return [];
    }
  }

  void _onSelectionChanged() {
    widget.onChanged(
      selectedYear,
      selectedMonth,
      selectedDay,
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("개봉 연월일", style: subtitleText),
        const SizedBox(height: 8),
        Row(
          children: [
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
                  selectedDay = null;
                  _onSelectionChanged();
                });
              },
            ),
            const SizedBox(width: 16),
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
                  selectedDay = null;
                  _onSelectionChanged();
                });
              },
            ),
            const SizedBox(width: 16),
            DropdownButton<int>(
              hint: const Text("일"),
              value: selectedDay,
              items:
                  days.map((day) {
                    return DropdownMenuItem<int>(
                      value: day,
                      child: Text("$day일"),
                    );
                  }).toList(),
              onChanged: (value) {
                setState(() {
                  selectedDay = value;
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
