import 'package:cricgenius/screens/home.dart';
import 'package:cricgenius/widgets/bottom_bar.dart';
import 'package:cricgenius/widgets/svg_icon.dart';
import 'package:flutter/material.dart';

class ScreenRouter extends StatefulWidget {
  const ScreenRouter({super.key});

  @override
  State<ScreenRouter> createState() => _ScreenRouterState();
}

class _ScreenRouterState extends State<ScreenRouter> {
  @override
  Widget build(BuildContext context) {
    List<Widget> tabList = [
      const HomeScreen(),
      const HomeScreen(),
      const HomeScreen(),
      const HomeScreen(),
    ];

    int currentIndex = 0;

    return Scaffold(
      resizeToAvoidBottomInset: false,
      // bottomNavigationBar: BottomBar(
      //   items: const [
      //     SvgIcon("assets/images/home.svg"),
      //     SvgIcon("assets/images/my-bets.svg"),
      //     SvgIcon("assets/images/scores.svg"),
      //     SvgIcon("assets/images/profile.svg")
      //   ],
      //   titles: const ["Home", "My Bets", "Scores", "Profile"],
      //   activeItems: const [
      //     SvgIcon("assets/images/home_active.svg"),
      //     SvgIcon("assets/images/my-bets.svg"),
      //     SvgIcon("assets/images/scores.svg"),
      //     SvgIcon("assets/images/profile.svg")
      //   ],
      //   currentIndex: 0,
      //   onTap: (value) {
      //     // currentIndex = value;
      //   },
      // ),
      body: Stack(
        children: [
          tabList[currentIndex],
        ],
      ),
    );
  }
}
