import 'package:flutter/material.dart';

// Button Widget
class LargeButton extends StatelessWidget {
  const LargeButton({
    super.key,
    required this.title,
    required this.onTap,
    this.color,
    this.borderRadius = 16,
    this.isDisable = false,
    this.isLoading = false,
  });

  final String title;
  final VoidCallback onTap;
  final bool isDisable;
  final bool isLoading;
  final double borderRadius;
  final Color? color;

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12),
      color: Colors.transparent,
      height: 60,
      child: Material(
        color: isDisable ? const Color.fromARGB(100, 160, 160, 160) : (color ?? const Color(0xff3A63ED)),
        borderRadius: BorderRadius.circular(borderRadius),
        child: InkWell(
          onTap: isDisable ? null : onTap,
          borderRadius: BorderRadius.circular(borderRadius),
          child: Center(
            child: isLoading
                ? const CircularProgressIndicator(
                    valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  )
                : Text(
                    title,
                    style: TextStyle(
                      color: isDisable
                          ? const Color.fromARGB(255, 150, 149, 149)
                          : Colors.white,
                      fontSize: 24,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
          ),
        ),
      ),
    );
  }
}
