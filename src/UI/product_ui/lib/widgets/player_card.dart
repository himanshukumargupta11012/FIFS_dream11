import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/model/player_prediction.dart';
import 'package:cricgenius/screens/player_info.dart';
import 'package:flutter/material.dart';

class PlayerCard extends StatelessWidget {
  const PlayerCard(
      {super.key,
      required this.player,
      required this.explainability,
      required this.score,
      this.isCaptain = false,
      this.isViceCaptain = false,
      this.bg = Colors.white,
      this.textColor = Colors.black,
      required this.team});

  final PlayerPredictionModel player;
  final Color bg;
  final Color textColor;
  final int score;
  final bool isViceCaptain;
  final bool isCaptain;
  final String explainability;
  final String team;

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    return Stack(children: [
      Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          CircleAvatar(
            radius: 40,
            backgroundImage: NetworkImage(player.getPlayerImage()),
          ),
          Container(
            constraints: BoxConstraints(maxWidth: (screenWidth - 50) / 3),
            decoration: BoxDecoration(
              color: bg,
              borderRadius: BorderRadius.circular(2),
            ),
            padding: const EdgeInsets.symmetric(vertical: 2, horizontal: 4),
            child: Text(
              player.name,
              style: TextStyle(
                overflow: TextOverflow.ellipsis,
                color: textColor,
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
          ),
        ],
      ),
      isCaptain || isViceCaptain
          ? Positioned(
              bottom: isCaptain ? 20.5 : 25,
              right: 2,
              child: Container(
                decoration: BoxDecoration(
                  shape: BoxShape.circle,

                  border: Border.all(
                      width: 1, color: Colors.white), // Outer white border
                ),
                child: Container(
                  decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      border: Border.all(width: 2, color: Color(0xFF3A63ED))),
                  padding: EdgeInsets.all(isCaptain ? 5 : 3.5),
                  child: Text(
                    isCaptain
                        ? "C"
                        : isViceCaptain
                            ? "VC"
                            : "",
                    style: TextStyle(
                        fontSize: isCaptain ? 12 : 10,
                        color: Color(0xFF3A63ED),
                        fontWeight:
                            isCaptain ? FontWeight.w600 : FontWeight.w700),
                  ),
                ),
              ),
            )
          : Positioned(
              bottom: -2,
              right: -2,
              child: Container(),
            ),
      Positioned(
        top: -2,
        right: -2,
        child: GestureDetector(
          onTap: () {
            Navigator.push(
              context,
              MaterialPageRoute(
                  builder: (context) => PlayerInfoScreen(
                        team: team,
                        player: PlayerModel(
                            name: player.name,
                            uniqueIdentifier: player.uniqueIdentifier),
                        explainability: explainability,
                        score: score,
                        isCaptain: isCaptain,
                        isViceCaptain: isViceCaptain,
                      )),
            );
          },
          child: Container(
            decoration: const BoxDecoration(
              color: Colors.black54,
              shape: BoxShape.circle,
            ),
            padding: const EdgeInsets.all(1),
            child: const Icon(
              Icons.info_outline,
              color: Colors.white,
              size: 20,
            ),
          ),
        ),
      ),
    ]);
  }
}
