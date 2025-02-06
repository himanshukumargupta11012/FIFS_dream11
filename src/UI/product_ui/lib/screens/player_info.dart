import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/model/upcoming.dart';
import 'package:cricgenius/services/api_services.dart';
import 'package:cricgenius/widgets/appbar.dart';
import 'package:flutter/material.dart';
import 'package:flutter_tts/flutter_tts.dart';

class PlayerInfoScreen extends StatefulWidget {
  final PlayerModel player;
  final int score;
  final String explainability;
  final bool isViceCaptain;
  final bool isCaptain;
  final String team;

  const PlayerInfoScreen({
    Key? key,
    required this.player,
    required this.score,
    required this.explainability,
    required this.team,
    this.isCaptain = false,
    this.isViceCaptain = false,
  }) : super(key: key);

  @override
  State<PlayerInfoScreen> createState() => _PlayerInfoScreenState();
}

enum TtsState { playing, stopped, paused, continued }

class _PlayerInfoScreenState extends State<PlayerInfoScreen> {
  PlayerModel? player;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    fetchPlayerData();
  }

  Future<void> fetchPlayerData() async {
    try {
      final fetchedPlayer =
          await ApiServices().getPlayerInfo(widget.player.uniqueIdentifier);
      setState(() {
        player = fetchedPlayer;
        isLoading = false;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      print("Error fetching player data: $e");
    }
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;
    FlutterTts flutterTts = FlutterTts();
    bool isPlaying = false;

    return Scaffold(
      floatingActionButton: Container(
        padding: const EdgeInsets.all(8),
        decoration: const BoxDecoration(
          color: Color(0xFF3A63ED),
          shape: BoxShape.circle,
        ),
        child: IconButton(
            onPressed: () async {
              if (isPlaying) {
                var result = await flutterTts.stop();
                if (result == 1) {
                  isPlaying = false;
                }
              } else {
                var result = await flutterTts.speak(widget.explainability);
                if (result == 1) setState(() => isPlaying = true);
              }
            },
            icon: const Icon(Icons.volume_up, size: 36, color: Colors.white)),
      ),
      backgroundColor: Colors.white,
      body: isLoading
          ? Center(child: CircularProgressIndicator())
          : Column(
              children: [
                const CustomAppBar(),
                Padding(
                  padding: EdgeInsets.all(16),
                  child: Row(
                    children: [
                      CircleAvatar(
                        radius: 50,
                        backgroundImage:
                            NetworkImage(widget.player.getPlayerImage()),
                      ),
                      SizedBox(
                        width: 10,
                      ),
                      Expanded(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.start,
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              widget.player.name,
                              softWrap: true,
                              maxLines: 2,
                              textAlign: TextAlign.start,
                              style: const TextStyle(
                                  height: 1.2,
                                  fontSize: 24,
                                  fontWeight: FontWeight.w600),
                            ),
                            player != null && player!.playingRole != null
                                ? Container(
                                    margin: EdgeInsets.symmetric(vertical: 5),
                                    padding: EdgeInsets.symmetric(
                                        horizontal: 15, vertical: 5),
                                    decoration: BoxDecoration(
                                      color: Color(0xFF203591),
                                      borderRadius: BorderRadius.circular(10),
                                    ),
                                    child: Text(
                                      player!.playingRole!,
                                      style: TextStyle(
                                          color: Colors.white, fontSize: 16),
                                    ),
                                  )
                                : SizedBox(),
                            Text(
                              'Team: ${widget.team}',
                              overflow: TextOverflow.ellipsis,
                              textAlign: TextAlign.start,
                              style: const TextStyle(
                                  fontSize: 18, fontWeight: FontWeight.w500),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    children: [
                      Container(
                        height: 1,
                        width: screenWidth - 20,
                        decoration: const BoxDecoration(
                          color: Color.fromARGB(255, 51, 51, 51),
                        ),
                      ),
                      const SizedBox(
                        height: 20,
                      ),
                      Container(
                        padding: const EdgeInsets.all(20),
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(20),
                          color: Colors.white,
                          border: Border.all(
                            color: Colors.grey,
                            width: 0.5,
                          ),
                          boxShadow: const [
                            BoxShadow(
                              color: Color(0x3F000000),
                              blurRadius: 14.80,
                              offset: Offset(0, 6),
                              spreadRadius: -9,
                            )
                          ],
                        ),
                        child: Row(
                          children: [
                            Expanded(
                              child: Column(
                                children: [
                                  Text(
                                    'AI Score',
                                    textAlign: TextAlign.center,
                                    style: TextStyle(
                                        color: Colors.grey,
                                        fontSize: 16,
                                        height: 0.9,
                                        fontWeight: FontWeight.w500),
                                  ),
                                  SizedBox(
                                    height: 15,
                                  ),
                                  Text(
                                    widget.isCaptain || widget.isViceCaptain
                                        ? (widget.score * 2).toString()
                                        : (widget.score).toString(),
                                    style: TextStyle(
                                        color: Colors.black,
                                        fontSize: 24,
                                        fontWeight: FontWeight.w600),
                                  ),
                                ],
                              ),
                            ),
                            Container(
                              height: 80,
                              width: 1.5,
                              decoration: const BoxDecoration(
                                color: Colors.grey,
                              ),
                            ),
                            if (player != null)
                              Expanded(
                                flex: 2,
                                child: Padding(
                                  padding: const EdgeInsets.symmetric(
                                      horizontal: 15),
                                  child: Column(
                                    children: [
                                      player!.battingStyle != null
                                          ? Row(
                                              children: [
                                                Image.asset(
                                                  'assets/images/bat.png',
                                                  width: 25,
                                                ),
                                                const SizedBox(
                                                  width: 10,
                                                ),
                                                Flexible(
                                                    child: Text(
                                                  player!.battingStyle!,
                                                  maxLines: 2,
                                                )),
                                              ],
                                            )
                                          : SizedBox(),
                                      const SizedBox(
                                        height: 10,
                                      ),
                                      player!.bowlingStyle != null
                                          ? Row(
                                              children: [
                                                Image.asset(
                                                  'assets/images/ball.png',
                                                  width: 25,
                                                ),
                                                const SizedBox(
                                                  width: 10,
                                                ),
                                                Flexible(
                                                  child: Text(
                                                    player!.bowlingStyle!,
                                                    maxLines: 2,
                                                  ),
                                                ),
                                              ],
                                            )
                                          : SizedBox(),
                                    ],
                                  ),
                                ),
                              )
                            else
                              SizedBox(),
                          ],
                        ),
                      ),
                      const SizedBox(
                        height: 20,
                      ),
                      Text(
                        widget.explainability,
                        style: TextStyle(
                            fontSize: 20, fontWeight: FontWeight.normal),
                      ),
                    ],
                  ),
                )
              ],
            ),
    );
  }
}
