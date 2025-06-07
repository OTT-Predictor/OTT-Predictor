import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class SummaryInput extends StatefulWidget {
  final Function(String) onChanged;

  const SummaryInput({
    super.key,
    required this.onChanged,
  });

  @override
  State<SummaryInput> createState() =>
      _SummaryInputState();
}

class _SummaryInputState
    extends State<SummaryInput> {
  final TextEditingController _controller =
      TextEditingController();

  /// 간단한 XSS 필터링: <태그> 제거
  String sanitize(String input) {
    return input.replaceAll(
      RegExp(r'<[^>]*>'),
      '',
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("줄거리 요약", style: subtitleText),
        const SizedBox(height: 8),
        TextField(
          controller: _controller,
          maxLines: 6,
          maxLength: 500,
          keyboardType: TextInputType.multiline,
          decoration: const InputDecoration(
            hintText: '줄거리를 500자 이내로 입력해주세요',
            border: OutlineInputBorder(),
          ),
          onChanged: (value) {
            final clean = sanitize(value);
            widget.onChanged(
              clean,
            ); // 클린된 내용 외부로 전달
          },
        ),
      ],
    );
  }
}
