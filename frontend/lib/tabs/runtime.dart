import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:ossw4_msps/main.dart';

class RuntimeInput extends StatefulWidget {
  final Function(int?) onChanged;

  const RuntimeInput({
    super.key,
    required this.onChanged,
  });

  @override
  State<RuntimeInput> createState() =>
      _RuntimeInputState();
}

class _RuntimeInputState
    extends State<RuntimeInput> {
  final TextEditingController _controller =
      TextEditingController();
  String? _error;

  void _handleInput(String value) {
    final numeric = int.tryParse(value);

    if (numeric == null) {
      setState(() {
        _error = "숫자만 입력해주세요";
      });
      widget.onChanged(null);
      return;
    }

    if (numeric < 1 || numeric > 500) {
      setState(() {
        _error = "1~500 사이의 숫자를 입력해주세요";
      });
      widget.onChanged(null);
    } else {
      setState(() {
        _error = null;
      });
      widget.onChanged(numeric);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("상영 시간", style: subtitleText),
        const SizedBox(height: 8),
        TextField(
          controller: _controller,
          keyboardType: TextInputType.number,
          inputFormatters: [
            TextInputFormatter.withFunction((
              oldValue,
              newValue,
            ) {
              final digitsOnly = newValue.text
                  .replaceAll(
                    RegExp(r'[^\d]'),
                    '',
                  );
              return newValue.copyWith(
                text: digitsOnly,
                selection:
                    TextSelection.collapsed(
                      offset: digitsOnly.length,
                    ),
              );
            }),
          ],
          onChanged: _handleInput,
          decoration: InputDecoration(
            hintText: '예: 120',
            suffixText: '분',
            suffixIcon: IconButton(
              icon: const Icon(Icons.close),
              onPressed: () {
                _controller.clear();
                setState(() {});
              },
            ),
            errorText: _error,
            border: const OutlineInputBorder(),
          ),
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
