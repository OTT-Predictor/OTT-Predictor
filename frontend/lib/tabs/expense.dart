import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';
import 'package:flutter/services.dart';

class ProductionBudgetInput
    extends StatefulWidget {
  final Function(int?) onChanged;

  const ProductionBudgetInput({
    super.key,
    required this.onChanged,
  });

  @override
  State<ProductionBudgetInput> createState() =>
      _ProductionBudgetInputState();
}

class _ProductionBudgetInputState
    extends State<ProductionBudgetInput> {
  final TextEditingController _controller =
      TextEditingController();

  void _handleInput(String value) {
    final numeric = int.tryParse(
      value.replaceAll(',', ''),
    );
    widget.onChanged(numeric);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        const Text("제작비", style: subtitleText),
        const SizedBox(height: 8),
        TextField(
          controller: _controller,
          keyboardType: TextInputType.number,
          inputFormatters: [
            TextInputFormatter.withFunction((
              oldValue,
              newValue,
            ) {
              final newText = newValue.text;
              final digitsOnly = newText
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
          ], //숫자만
          onChanged: _handleInput,
          decoration: InputDecoration(
            hintText: '예: 100000000',
            suffixText: '원',
            suffixIcon: IconButton(
              icon: Icon(Icons.close),
              onPressed: () {
                _controller.clear();
                setState(() {});
              },
            ),
            border: OutlineInputBorder(),
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
