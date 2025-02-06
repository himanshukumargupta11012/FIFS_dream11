import 'dart:ui';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';


class BottomBar extends StatefulWidget {
  const BottomBar(
      {super.key,
      required this.items,
      required this.titles,
      this.showTitle = true,
      this.midGap = 20,
      required this.activeItems,
      this.currentIndex = 0,
      this.color = const Color(0xff007BFF),
      this.onTap});
  final List<Widget> items;
  final List<Widget> activeItems;
  final List<String> titles;
  final bool showTitle;
  final double midGap;
  final int currentIndex;
  final Color color;
  final ValueChanged<int>? onTap;

  @override
  State<BottomBar> createState() => _BottomBarState();
}

class _BottomBarState extends State<BottomBar> {
  int activeIndex = 0;

  @override
  void initState() {
    super.initState();
    activeIndex = widget.currentIndex;
  }

  changeIndex(int index) {
    // setState(() {
    //   activeIndex = index;
    // });
    // if (widget.onTap != null) {
    //   widget.onTap!(index);
    // }
  }

  @override
  Widget build(BuildContext context) {
    double width = MediaQuery.of(context).size.width;
    return Container(
      // margin: const EdgeInsets.only(left: 6, right: 6, bottom: 24),
      child: Stack(
        alignment: Alignment.center,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: BackdropFilter(
              filter: ImageFilter.blur(sigmaX: 12.0, sigmaY: 12.0),
              child: Container(
                width: width,
                height: 65,
                decoration: const BoxDecoration(
                  color: Color.fromARGB(255, 245, 245, 245),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: listOfItems(),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget iconWidgets(int index) {
    if (index == activeIndex) {
      return activeIconWidgets(index);
    }

    return InkWell(
      onTap: () {
        changeIndex(index);
      },
      highlightColor: Colors.transparent,
      splashColor: Colors.transparent,
      child: Padding(
        padding: const EdgeInsets.only(top: 8, bottom: 8, left: 4, right: 4),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            widget.items[index],
            if (widget.showTitle)
              Text(
                widget.titles[index],
                style: GoogleFonts.montserrat(
                    fontSize: 12, fontWeight: FontWeight.w500),
              ),
          ],
        ),
      ),
    );
  }

  Widget activeIconWidgets(int index) {
    return Padding(
      padding: const EdgeInsets.only(top: 8, bottom: 8, left: 4, right: 4),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          widget.activeItems[index],
          if (widget.showTitle)
            Text(
              widget.titles[index],
              style: GoogleFonts.montserrat(
                  fontSize: 12,
                  fontWeight: FontWeight.w500,
                  color: widget.color),
            ),
        ],
      ),
    );
  }

  List<Widget> listOfItems() {
    return [
      for (int i = 0; i < widget.items.length; i++)
        // if (i == widget.items.length ~/ 2) ...[
        //   SizedBox(width: widget.midGap),
        //   iconWidgets(i),
        // ] else
        iconWidgets(i),
    ];
  }
}
