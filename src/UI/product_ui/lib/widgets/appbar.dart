import 'package:cricgenius/widgets/svg_icon.dart';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

class CustomAppBar extends StatelessWidget {
  const CustomAppBar({super.key});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      height: 94 + 24,
      width: MediaQuery.of(context).size.width,
      child: Stack(
        children: [
          Container(
            decoration: const ShapeDecoration(
              gradient: LinearGradient(
                begin: Alignment(1.00, 0.00),
                end: Alignment(-1, 0),
                colors: [Color(0xFF2646B1), Color(0xFF007BFF)],
              ),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.only(
                  bottomLeft: Radius.circular(10),
                  bottomRight: Radius.circular(10),
                ),
              ),
              shadows: [
                BoxShadow(
                  color: Color(0x3F000000),
                  blurRadius: 6.60,
                  offset: Offset(0, 0),
                  spreadRadius: 0,
                )
              ],
            ),
          ),
          Center(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SvgIcon("assets/images/logo.svg"),
                const SizedBox(width: 4,),
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    "CricGenius",
                    style: GoogleFonts.kronaOne(color: Colors.white, fontSize: 28),
                  ),
                )
              ],
            ),
          )
        ],
      ),
    );
  }
}
