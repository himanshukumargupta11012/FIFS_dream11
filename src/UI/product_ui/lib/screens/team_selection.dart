import 'package:cricgenius/model/player_prediction.dart';
import 'package:cricgenius/model/team.dart';
import 'package:cricgenius/widgets/country_card.dart';
import 'package:cricgenius/widgets/player_card.dart';
import 'package:cricgenius/widgets/player_selection_card.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class TeamSelectionScreen extends StatefulWidget {
  final List<PlayerPredictionModel> playerPredictions;
  final TeamModel teamA;
  final TeamModel teamB;

  const TeamSelectionScreen(
      {super.key,
      required this.playerPredictions,
      required this.teamA,
      required this.teamB});

  @override
  State<TeamSelectionScreen> createState() => _TeamSelectionScreenState();
}

class _TeamSelectionScreenState extends State<TeamSelectionScreen> {
  // final List<Map<String, String>> players = [
  //   {
  //     "name": "Virat Kohli",
  //     "role": "Batsman",
  //     "image":
  //         "https://upload.wikimedia.org/wikipedia/commons/9/9b/Virat_Kohli_in_PMO_New_Delhi.jpg",
  //     "stats": "Highest Score: 84\nHS Against Pakistan: 84"
  //   },
  //   {
  //     "name": "MS Dhoni",
  //     "role": "Wicket Keeper",
  //     "image":
  //         "https://assets.vogue.in/photos/5f3a37acac1b7909f36d6814/1:1/w_3456,h_3456,c_limit/Mahendra%20Singh%20Dhoni%20fun%20facts.jpg",
  //     "stats": "Highest Score: 183\nHS Against Sri Lanka: 183"
  //   },
  //   {
  //     "name": "Rishabh Pant",
  //     "role": "Wicket Keeper",
  //     "image":
  //         "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/323000/323036.1.png",
  //     "stats": "Highest Score: 183\nHS Against Sri Lanka: 183"
  //   },
  //   {
  //     "name": "Ravichandran Ashwin",
  //     "role": "Wicket Keeper",
  //     "image":
  //         "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/316500/316521.2.png",
  //     "stats": "Highest Score: 183\nHS Against Sri Lanka: 183"
  //   },
  //   {
  //     "name": "Devdutt Padikkal",
  //     "role": "Wicket Keeper",
  //     "image":
  //         "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/322200/322229.1.png",
  //     "stats": "Highest Score: 183\nHS Against Sri Lanka: 183"
  //   },
  //   {
  //     "name": "Rohit Sharma",
  //     "role": "Wicket Keeper",
  //     "image":
  //         "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/385800/385819.2.png",
  //     "stats": "Highest Score: 183\nHS Against Sri Lanka: 183"
  //   },
  //   {
  //     "name": "Kuldeep Yadav",
  //     "role": "Wicket Keeper",
  //     "image":
  //         "https://img1.hscicdn.com/image/upload/f_auto,t_h_100/lsci/db/PICTURES/CMS/319900/319943.2.png",
  //     "stats": "Highest Score: 183\nHS Against Sri Lanka: 183"
  //   },
  // ];

  int currentIndex = 0;
  final _controller = PageController(viewportFraction: 0.55);

  void onSwipe(int index) {
    setState(() {
      currentIndex = index % widget.playerPredictions.length;
    });
    _controller.animateToPage(index,
        duration: const Duration(milliseconds: 450),
        curve: Curves.linearToEaseOut);
  }

  getTeamNameOfAPlayer(String uniqueIdentifier) {
    if (widget.teamA.players.any((p) {
      return p.uniqueIdentifier == uniqueIdentifier;
    })) {
      return widget.teamA.teamName;
    }
    return widget.teamB.teamName;
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
                const Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    CountryCard(
                      countryName: "IND",
                      flagUrl:
                          "https://upload.wikimedia.org/wikipedia/en/thumb/4/41/Flag_of_India.svg/800px-Flag_of_India.svg.png?20240827082344",
                      isLeft: true,
                    ),
                    CountryCard(
                      countryName: "SA",
                      flagUrl:
                          "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSTi2o0h9Z1qyaZBR2NBOKv76yhM6m9zBdvMA&s",
                      isLeft: false,
                    ),
                  ],
                ),
                const SizedBox(
                  height: 10,
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
                                    widget.playerPredictions[0].uniqueIdentifier),
                                player: widget.playerPredictions[0],
                                score: widget.playerPredictions[0].score,
                                explainability:
                                    widget.playerPredictions[0].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    widget.playerPredictions[1].uniqueIdentifier),
                                player: widget.playerPredictions[1],
                                explainability:
                                    widget.playerPredictions[1].explain,
                                score: widget.playerPredictions[1].score,
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
                                    widget.playerPredictions[2].uniqueIdentifier),
                                player: widget.playerPredictions[2],
                                score: widget.playerPredictions[2].score,
                                explainability:
                                    widget.playerPredictions[2].explain,
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
                                    widget.playerPredictions[3].uniqueIdentifier),
                                player: widget.playerPredictions[3],
                                score: widget.playerPredictions[3].score,
                                explainability:
                                    widget.playerPredictions[3].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    widget.playerPredictions[4].uniqueIdentifier),
                                player: widget.playerPredictions[4],
                                score: widget.playerPredictions[4].score,
                                explainability:
                                    widget.playerPredictions[4].explain,
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
                                    widget.playerPredictions[5].uniqueIdentifier),
                                player: widget.playerPredictions[5],
                                explainability:
                                    widget.playerPredictions[5].explain,
                                score: widget.playerPredictions[5].score,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    widget.playerPredictions[6].uniqueIdentifier),
                                player: widget.playerPredictions[6],
                                score: widget.playerPredictions[6].score,
                                explainability:
                                    widget.playerPredictions[6].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    widget.playerPredictions[7].uniqueIdentifier),
                                player: widget.playerPredictions[7],
                                explainability:
                                    widget.playerPredictions[7].explain,
                                score: widget.playerPredictions[7].score,
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
                                    widget.playerPredictions[8].uniqueIdentifier),
                                player: widget.playerPredictions[8],
                                score: widget.playerPredictions[8].score,
                                explainability:
                                    widget.playerPredictions[8].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    widget.playerPredictions[9].uniqueIdentifier),
                                player: widget.playerPredictions[9],
                                score: widget.playerPredictions[9].score,
                                explainability:
                                    widget.playerPredictions[9].explain,
                              ),
                              PlayerCard(
                                team: getTeamNameOfAPlayer(
                                    widget.playerPredictions[10].uniqueIdentifier),
                                player: widget.playerPredictions[10],
                                score: widget.playerPredictions[10].score,
                                explainability:
                                    widget.playerPredictions[10].explain,
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
                Expanded(
                  child: PageView.builder(
                    itemCount: widget.playerPredictions.length,
                    controller: PageController(viewportFraction: 0.55),
                    onPageChanged: onSwipe,
                    hitTestBehavior: HitTestBehavior.translucent,
                    itemBuilder: (context, index) {
                      final player = widget.playerPredictions[
                          index % widget.playerPredictions.length];
                      final isSelected = index == currentIndex;
                      final double verticalOffset = isSelected ? -20 : 20.0;
                      final double horizontalOffset = isSelected
                          ? 0.0
                          : index == currentIndex - 1
                              ? 30
                              : -30;

                      return Transform.translate(
                        offset: Offset(horizontalOffset, verticalOffset),
                        child: Transform.scale(
                          scale: isSelected ? 0.9 : 0.7,
                          child: PlayerSelectionCard(
                            name: player.name,
                            // role: player["role"]!,
                            imageUrl: player.getPlayerImage(),
                            // stats: player.,
                            isSelected: isSelected,
                          ),
                        ),
                      );
                    },
                  ),
                ),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                  children: [
                    ElevatedButton(
                      onPressed: () {
                        // Handle AI Team functionality
                      },
                      style: ElevatedButton.styleFrom(
                        shape: const CircleBorder(
                          side: BorderSide(color: Colors.white, width: 3),
                        ),
                        padding: const EdgeInsets.all(2),
                        backgroundColor: const Color(0xFF3A63ED),
                      ),
                      child: const Icon(
                        CupertinoIcons.add_circled,
                        size: 85,
                        color: Colors.white,
                      ),
                    ),
                  ],
                ),
                const SizedBox(
                  height: 10,
                )
              ],
            ),
          ),
        ],
      ),
    );
  }
}
