import 'package:cricgenius/model/player_prediction.dart';
import 'package:cricgenius/model/team.dart';
import 'package:cricgenius/screens/team_selection.dart';
import 'package:cricgenius/widgets/country_card.dart';
import 'package:cricgenius/widgets/player_card.dart';
import 'package:cricgenius/widgets/svg_icon.dart';
import 'package:flutter/material.dart';

class AiTeamScreen extends StatelessWidget {
  final List<PlayerPredictionModel> playerPredictions;
  final TeamModel teamA, teamB;

  const AiTeamScreen({
    super.key,
    required this.playerPredictions,
    required this.teamA,
    required this.teamB,
  });

  getTeamNameOfAPlayer(String uniqueIdentifier) {
    if (teamA.players.any((p) {
      return p.uniqueIdentifier == uniqueIdentifier;
    })) {
      return teamA.teamName;
    }
    return teamB.teamName;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: Image.asset(
              'assets/images/cricket_bg.png',
              fit: BoxFit.fitHeight,
            ),
          ),
          SafeArea(
            child: Column(
              children: [
                const SizedBox(
                  height: 10,
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CountryCard(
                      countryName: teamA.getAbbreviation().toUpperCase(),
                      flagUrl: teamA.getTeamLogo(),
                      isLeft: true,
                    ),
                    const Expanded(
                      child: Padding(
                          padding: EdgeInsets.fromLTRB(0, 20, 0, 0),
                          child: SvgIcon("assets/images/ai_banner.svg")),
                    ),
                    CountryCard(
                      countryName: teamB.getAbbreviation(),
                      flagUrl: teamB.getTeamLogo(),
                      isLeft: false,
                    ),
                  ],
                ),
                const SizedBox(
                  height: 0,
                ),
                Container(
                  margin: EdgeInsets.fromLTRB(0, 5, 0, 15),
                  padding: EdgeInsets.symmetric(vertical: 5, horizontal: 15),
                  decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(10)),
                  child: Text(
                    "AI Score: ${playerPredictions.asMap().entries.fold(0, (total, entry) {
                      int index = entry.key;
                      var player = entry.value;
                      int score = player.score;
                      if (index < 2) {
                        score *= 2;
                      }
                      return total + score;
                    })}",
                    style: TextStyle(
                        color: Color(0xFF3A63ED),
                        fontWeight: FontWeight.w600,
                        fontSize: 22),
                  ),
                ),
                Expanded(
                  child: SingleChildScrollView(
                    child: Column(
                      children: [
                        Padding(
                          padding: EdgeInsets.fromLTRB(10, 0, 10, 10),
                          child: Wrap(
                            alignment: WrapAlignment.center,
                            runSpacing: 10,
                            spacing: 10,
                            children: [
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[0].uniqueIdentifier),
                                player: playerPredictions[0],
                                score: playerPredictions[0].score,
                                explainability: playerPredictions[0].explain,
                                isCaptain: true,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[1].uniqueIdentifier),
                                player: playerPredictions[1],
                                explainability: playerPredictions[1].explain,
                                score: playerPredictions[1].score,
                                isViceCaptain: true,
                              ),
                            ],
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.fromLTRB(10, 0, 10, 10),
                          child: Wrap(
                            alignment: WrapAlignment.center,
                            runSpacing: 10,
                            spacing: 10,
                            children: [
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[2].uniqueIdentifier),
                                player: playerPredictions[2],
                                score: playerPredictions[2].score,
                                explainability: playerPredictions[2].explain,
                              ),
                            ],
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.fromLTRB(10, 0, 10, 10),
                          child: Wrap(
                            alignment: WrapAlignment.center,
                            runSpacing: 10,
                            spacing: 10,
                            children: [
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[3].uniqueIdentifier),
                                player: playerPredictions[3],
                                score: playerPredictions[3].score,
                                explainability: playerPredictions[3].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[4].uniqueIdentifier),
                                player: playerPredictions[4],
                                score: playerPredictions[4].score,
                                explainability: playerPredictions[4].explain,
                              ),
                            ],
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.fromLTRB(10, 0, 10, 10),
                          child: Wrap(
                            alignment: WrapAlignment.center,
                            runSpacing: 10,
                            spacing: 10,
                            children: [
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[5].uniqueIdentifier),
                                player: playerPredictions[5],
                                explainability: playerPredictions[5].explain,
                                score: playerPredictions[5].score,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[6].uniqueIdentifier),
                                player: playerPredictions[6],
                                score: playerPredictions[6].score,
                                explainability: playerPredictions[6].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[7].uniqueIdentifier),
                                player: playerPredictions[7],
                                explainability: playerPredictions[7].explain,
                                score: playerPredictions[7].score,
                              ),
                            ],
                          ),
                        ),
                        Padding(
                          padding: EdgeInsets.fromLTRB(10, 0, 10, 10),
                          child: Wrap(
                            alignment: WrapAlignment.center,
                            runSpacing: 10,
                            spacing: 10,
                            children: [
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[8].uniqueIdentifier),
                                player: playerPredictions[8],
                                score: playerPredictions[8].score,
                                explainability: playerPredictions[8].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[9].uniqueIdentifier),
                                player: playerPredictions[9],
                                score: playerPredictions[9].score,
                                explainability: playerPredictions[9].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    playerPredictions[10].uniqueIdentifier),
                                player: playerPredictions[10],
                                score: playerPredictions[10].score,
                                explainability: playerPredictions[10].explain,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(
                  height: 10,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
