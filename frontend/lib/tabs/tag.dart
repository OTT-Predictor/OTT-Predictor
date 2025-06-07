import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class KeywordInput extends StatefulWidget {
  final Function(List<String>) onChanged;

  const KeywordInput({
    super.key,
    required this.onChanged,
  });

  @override
  State<KeywordInput> createState() =>
      _KeywordInputState();
}

class _KeywordInputState
    extends State<KeywordInput> {
  final TextEditingController _controller =
      TextEditingController();
  final List<String> _keywords = [];

  void _tryAddKeyword(String value) {
    final parts = value.split(
      RegExp(r'[,\n]'),
    ); // 쉼표나 Enter(\n) 기준 분리
    for (var part in parts) {
      final keyword = part.trim();
      if (keyword.isNotEmpty &&
          !_keywords.contains(keyword)) {
        setState(() {
          _keywords.add(keyword);
        });
      }
    }
    _controller.clear();
    widget.onChanged(_keywords); // 외부 전달
  }

  void _removeKeyword(String keyword) {
    setState(() {
      _keywords.remove(keyword);
    });
    widget.onChanged(_keywords);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("키워드", style: subtitleText),
        const SizedBox(height: 8),
        TextField(
          controller: _controller,
          decoration: InputDecoration(
            hintText: 'Enter 또는 쉼표로 키워드를 추가하세요',
            border: const OutlineInputBorder(),
          ),
          onChanged: (value) {
            if (value.endsWith(',') ||
                value.endsWith('\n')) {
              _tryAddKeyword(value);
            }
          },
          onSubmitted: _tryAddKeyword,
        ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children:
              _keywords.map((keyword) {
                return Chip(
                  label: Text(keyword),
                  deleteIcon: const Icon(
                    Icons.close,
                  ),
                  onDeleted:
                      () =>
                          _removeKeyword(keyword),
                );
              }).toList(),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}
