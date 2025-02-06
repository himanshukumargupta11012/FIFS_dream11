import 'package:flutter/material.dart';

// Custom Dropdown to match the overall theme of the app
class CustomDropdown extends StatelessWidget {
  final List<String> list;
  final String hintText;
  final String? value;
  final ValueChanged<String?> onChanged;

  const CustomDropdown({
    super.key,

    // list of items shown in the dropdown as a list of String
    required this.list,
    required this.hintText,
    required this.value,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return DropdownButtonHideUnderline(
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
        decoration: BoxDecoration(
          color: const Color(0x1E020D07),
          borderRadius: BorderRadius.circular(16),
        ),
        child: DropdownButton<String>(
          isExpanded: true,
          value: value,
          items: list.map((String item) {
            return DropdownMenuItem<String>(
              value: item,
              child: Text(
                item,
                textAlign: TextAlign.start,
              ),
            );
          }).toList(),
          onChanged: onChanged,
          hint: Text(
            hintText,
            style: const TextStyle(
              color: Colors.black,
              fontSize: 16,
            ),
            textAlign: TextAlign.start,
          ),
          style: const TextStyle(
            color: Colors.black,
          ),
          icon: const Icon(Icons.arrow_drop_down, color: Color(0xFF7E7E7E)),
          dropdownColor: const Color(0xFFE0E0E0),
        ),
      ),
    );
  }
}

class CustomInputBox extends StatelessWidget {
  final TextEditingController controller;
  final String hintText;
  final TextInputType keyboardType;
  final bool obscureText;

  const CustomInputBox({
    super.key,
    required this.controller,
    required this.hintText,
    this.keyboardType = TextInputType.text,
    this.obscureText = false,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0x1E020D07),
        borderRadius: BorderRadius.circular(16),
      ),
      child: TextField(
        controller: controller,
        keyboardType: keyboardType,
        obscureText: obscureText,
        decoration: InputDecoration(
          hintText: hintText,
          border: InputBorder.none,
          hintStyle: const TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.normal,
            fontSize: 16,
          ),
        ),
        style: const TextStyle(
          color: Colors.black,
        ),
      ),
    );
  }
}

class CustomInputContainer extends StatelessWidget {
  final String title;
  final VoidCallback onPressed;

  const CustomInputContainer({
    super.key,
    required this.title,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onPressed,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
        decoration: BoxDecoration(
          color: const Color(0x1E020D07),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          title,
          style: const TextStyle(
            color: Colors.black,
            fontWeight: FontWeight.normal,
            fontSize: 16,
          ),
        ),
      ),
    );
  }
}
