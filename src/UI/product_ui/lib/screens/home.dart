import 'dart:convert';

import 'package:cricgenius/model/announcedMatch.dart';
import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/model/player_prediction.dart';
import 'package:cricgenius/model/team.dart';
import 'package:cricgenius/model/upcoming.dart';
import 'package:cricgenius/screens/ai_team.dart';
import 'package:cricgenius/services/api_services.dart';
import 'package:cricgenius/widgets/announced_match.dart';
import 'package:cricgenius/widgets/appbar.dart';
import 'package:cricgenius/widgets/large_button.dart';
import 'package:cricgenius/widgets/select_match_details.dart';
import 'package:cricgenius/widgets/select_players.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:intl/intl.dart';
import 'package:loading_animation_widget/loading_animation_widget.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen>
    with SingleTickerProviderStateMixin {
  final Set<String> selectedPlayersTeamA = {};
  final Set<String> selectedPlayersTeamB = {};
  // TextEditingController team1Controller = TextEditingController();
  // TextEditingController team2Controller = TextEditingController();
  List<PlayerModel> teamAPlayers = [];
  List<PlayerModel> teamBPlayers = [];
  List<PlayerModel> teamASelectedPlayers = [];
  List<PlayerModel> teamBSelectedPlayers = [];
  String teamAName = '';
  String teamBName = '';
  String matchId = '';

  String selectedDate = "";
  String selectedFormat = "";
  bool isMatchDetailsSelected = false;
  bool isLoaded = true;

  List<AnnouncedMatchModel> announcedMatches = [];
  List<Match> upcomingMatches = [];
  List<Match> pastMatches = [];
  bool isLoadingFirstTab = true;

  @override
  void initState() {
    super.initState();
    SystemChrome.setSystemUIOverlayStyle(const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.light,
        systemNavigationBarIconBrightness: Brightness.light));
    _tabController = TabController(length: 2, vsync: this);
    getMatches();
  }

  void showError(String msg) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(msg),
        duration: const Duration(milliseconds: 1500),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  void onChange(String name, bool isSelected, int team) {
    setState(() {
      final selectedPlayers =
          team == 0 ? selectedPlayersTeamA : selectedPlayersTeamB;
      if (isSelected) {
        selectedPlayers.remove(name);
      } else {
        if (selectedPlayers.length >= 11) {
          showError("You can only select up to 11 players for one team.");
        } else {
          selectedPlayers.add(name);
        }
      }
    });
  }

  void updateMatchDetailsSelection() async {
    setState(() {
      isMatchDetailsSelected = selectedDate != "" && selectedFormat.isNotEmpty;
    });

    if (isMatchDetailsSelected) {
      setState(() {
        isLoaded = false;
      });

      DateTime parsedDate;

      try {
        parsedDate = DateFormat('d/M/yyyy').parse(selectedDate);
      } catch (e) {
        debugPrint('Invalid date format: $selectedDate');
        setState(() {
          isLoaded = true;
        });
        return;
      }

      // Format the date to 'yyyy-MM-dd' with leading zeros
      final formattedDate = DateFormat('yyyy-MM-dd').format(parsedDate);
      debugPrint('Formatted date: $formattedDate');

      final resp = await ApiServices().getAnnouncedMatches(
        formattedDate,
        selectedFormat,
      );

      setState(() {
        announcedMatches = resp;
        showAnnouncedSheet = true;
        isLoaded = true;
      });
    }
  }

  bool showAnnouncedSheet = false;

  void openBottomSheet() {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      builder: (BuildContext context) {
        return SizedBox(
          height: MediaQuery.of(context).size.height * 0.5,
          child: Column(
            children: [
              const Padding(
                padding: EdgeInsets.all(16.0),
                child: Text(
                  "All Announced Matches",
                  style: TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 24,
                  ),
                ),
              ),
              const Divider(),
              Expanded(
                child: ListView.builder(
                  itemCount: announcedMatches.length,
                  itemBuilder: (BuildContext context, int index) {
                    final match = announcedMatches[index];
                    return AnnouncedMatch(
                      teamA: match.team1,
                      teamB: match.team2,
                      matchFormat: match.matchFormat,
                      matchStadium: match.matchStadium,
                      matchDate: match.matchDate,
                      onTap: (team1, team2, d, f) async {
                        await updateTeam(team1, team2);
                        teamASelectedPlayers.clear();
                        teamBSelectedPlayers.clear();

                        for (var matchPlayer in match.teamAPlayers) {
                          var matchingPlayer = teamAPlayers.firstWhere(
                            (player) =>
                                player.uniqueIdentifier ==
                                matchPlayer.uniqueIdentifier,
                            orElse: () => const PlayerModel(
                                name: '', uniqueIdentifier: ''),
                          );

                          if (matchingPlayer.name.isNotEmpty) {
                            teamASelectedPlayers.add(matchingPlayer);
                          }
                        }

                        for (var matchPlayer in match.teamBPlayers) {
                          var matchingPlayer = teamBPlayers.firstWhere(
                            (player) =>
                                player.uniqueIdentifier ==
                                matchPlayer.uniqueIdentifier,
                            orElse: () => const PlayerModel(
                                name: '', uniqueIdentifier: ''),
                          );

                          if (matchingPlayer.name.isNotEmpty) {
                            teamBSelectedPlayers.add(matchingPlayer);
                          }
                        }

                        final matchTeamAIdentifiers = {
                          for (var player in match.teamAPlayers)
                            player.uniqueIdentifier
                        };
                        final matchTeamBIdentifiers = {
                          for (var player in match.teamBPlayers)
                            player.uniqueIdentifier
                        };

                        teamAPlayers.sort((a, b) {
                          if (matchTeamAIdentifiers
                                  .contains(a.uniqueIdentifier) &&
                              !matchTeamAIdentifiers
                                  .contains(b.uniqueIdentifier)) {
                            return -1;
                          } else if (!matchTeamAIdentifiers
                                  .contains(a.uniqueIdentifier) &&
                              matchTeamAIdentifiers
                                  .contains(b.uniqueIdentifier)) {
                            return 1;
                          } else {
                            return 0;
                          }
                        });

                        teamBPlayers.sort((a, b) {
                          if (matchTeamBIdentifiers
                                  .contains(a.uniqueIdentifier) &&
                              !matchTeamBIdentifiers
                                  .contains(b.uniqueIdentifier)) {
                            return -1;
                          } else if (!matchTeamBIdentifiers
                                  .contains(a.uniqueIdentifier) &&
                              matchTeamBIdentifiers
                                  .contains(b.uniqueIdentifier)) {
                            return 1;
                          } else {
                            return 0;
                          }
                        });

                        setState(() {
                          matchId = announcedMatches[index].matchId;
                          showAnnouncedSheet = false; // Update the variable
                        });
                        Navigator.pop(context);
                        Navigator.pop(context);
                      },
                    );
                  },
                ),
              ),
            ],
          ),
        );
      },
    ).whenComplete(() {
      // Ensure the value is reset when the bottom sheet is dismissed
      setState(() {
        showAnnouncedSheet = false;
      });
    });
  }

  Future<void> updateTeam(String? teamA, String? teamB) async {
    setState(() {
      if (teamA != null) {
        teamAName = teamA;
      }

      if (teamB != null) {
        teamBName = teamB;
      }
    });

    if (teamAName.isEmpty || teamBName.isEmpty) {
      return;
    }

    setState(() {
      isLoaded = false;
    });

    // Simulate fetching players of both the teams
    // await Future.delayed(const Duration(seconds: 1));
    final t1 = await ApiServices().getTeamData(teamAName);
    final t2 = await ApiServices().getTeamData(teamBName);

    setState(() {
      teamAPlayers = t1?.players ?? [];
      teamBPlayers = t2?.players ?? [];
    });

    setState(() {
      isLoaded = true;
    });
  }

  List<Widget> tabList = [
    const HomeScreen(),
    const HomeScreen(),
    const HomeScreen(),
    const HomeScreen(),
  ];

  late TabController _tabController;

  Future<void> getMatches() async {
    try {
      final res = await ApiServices().getUpcomingMatches();

      String teamsJsonString =
          await rootBundle.loadString('assets/data/teams.json');
      Map<String, dynamic> teamsData = json.decode(teamsJsonString);

      if (res != null) {
        final matches = ApiResponse.fromJson(res, teamsData);
        setState(() {
          upcomingMatches = matches.matches
              .where((match) {
                return match.isUpcoming;
              })
              .toList()
              .reversed
              .toList();

          pastMatches = matches.matches.where((match) {
            return !match.isUpcoming && match.winner != "";
          }).toList();
        });
        setState(() {
          isLoadingFirstTab = false;
        });
      } else {
        setState(() {
          isLoadingFirstTab = false;
        });
      }
    } catch (e) {
      setState(() {
        isLoadingFirstTab = false;
      });
      debugPrint('Error fetching matches: $e');
    }
  }

  String _getMonthName(int month) {
    const months = [
      "January",
      "February",
      "March",
      "April",
      "May",
      "June",
      "July",
      "August",
      "September",
      "October",
      "November",
      "December"
    ];
    return months[month - 1];
  }

  String convertToAbbreviatedDate(String inputDate) {
    try {
      DateTime parsedDate = DateFormat('d MMMM yyyy').parse(inputDate);
      return DateFormat('d/M/yyyy').format(parsedDate);
    } catch (e) {
      return inputDate;
    }
  }

  onUpcomingMatchesCardClick(
      String team1, String team2, String date, String format) {
    setState(() {
      teamASelectedPlayers.clear();
      teamBSelectedPlayers.clear();
      selectedDate = convertToAbbreviatedDate(date);
      selectedFormat = format.toUpperCase();
      teamAName = team1;
      teamBName = team2;
    });
    _tabController.animateTo(1);

    updateTeam(team1, team2);
  }

  bool submitButtonClicked = false;

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    double screenHeight = MediaQuery.of(context).size.height;

    if (showAnnouncedSheet) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        openBottomSheet();
      });
    }

    return Scaffold(
      backgroundColor: Colors.white,
      body: !isLoaded
          ? const Center(child: CircularProgressIndicator())
          : Column(
              children: [
                const CustomAppBar(),
                SizedBox(
                  height: 45,
                  child: TabBar(
                    controller: _tabController,
                    labelColor: const Color(0xff428BE1),
                    unselectedLabelColor: Colors.black,
                    tabs: const [
                      Tab(
                        child: Text(
                          'Home',
                          style: TextStyle(fontSize: 15),
                        ),
                      ),
                      Tab(
                        child: Text(
                          'Select Custom Match',
                          style: TextStyle(fontSize: 15),
                        ),
                      ),
                    ],
                  ),
                ),
                // tab bar view here
                Expanded(
                  child: TabBarView(
                    controller: _tabController,
                    children: [
                      // first tab bar view widget
                      Container(
                        alignment: Alignment.topCenter,
                        child: isLoadingFirstTab
                            ? const Center(child: CircularProgressIndicator())
                            : upcomingMatches.isEmpty
                                ? Container(
                                    margin: EdgeInsets.only(top: 12),
                                    child: Text(
                                      "No Matches Data Found. Please Try Custom Match",
                                      style: TextStyle(fontSize: 14),
                                    ),
                                  )
                                : Column(
                                    children: [
                                      if (pastMatches.length > 0) ...[
                                        Container(
                                          width: double.infinity,
                                          padding: const EdgeInsets.only(
                                              left: 12, top: 12, bottom: 12),
                                          child: const Text(
                                            "Past Matches",
                                            style: TextStyle(
                                                fontWeight: FontWeight.bold,
                                                fontSize: 22),
                                          ),
                                        ),
                                        Container(
                                          height: 125,
                                          child: Expanded(
                                            child: ListView.builder(
                                              scrollDirection: Axis.horizontal,
                                              padding: EdgeInsets.all(0),
                                              itemCount: pastMatches.length,
                                              itemBuilder: (context, index) {
                                                final match =
                                                    pastMatches[index];
                                                return AnnouncedMatch(
                                                  winner: match.winner.isEmpty
                                                      ? null
                                                      : match.winner,
                                                  teamAFlag: match.team1Image,
                                                  teamBFlag: match.team2Image,
                                                  matchDate:
                                                      "${match.dateTimeGMT.toLocal().day} ${_getMonthName(match.dateTimeGMT.toLocal().month)} ${match.dateTimeGMT.toLocal().year}",
                                                  matchFormat: match.matchType,
                                                  teamA: match.team1,
                                                  teamB: match.team2,
                                                  matchStadium:
                                                      match.series.toString(),
                                                  onTap: (team1, team2, date,
                                                      format) {
                                                    onUpcomingMatchesCardClick(
                                                        team1,
                                                        team2,
                                                        date,
                                                        format);
                                                  },
                                                );
                                              },
                                            ),
                                          ),
                                        ),
                                      ],
                                      if (upcomingMatches.length > 0) ...[
                                        Container(
                                          width: double.infinity,
                                          padding: const EdgeInsets.only(
                                              left: 12, top: 12, bottom: 12),
                                          child: const Text(
                                            "Upcoming Matches",
                                            style: TextStyle(
                                                fontWeight: FontWeight.bold,
                                                fontSize: 22),
                                          ),
                                        ),
                                        Expanded(
                                          child: Container(
                                            child: ListView.builder(
                                              padding: EdgeInsets.all(0),
                                              itemCount: upcomingMatches.length,
                                              itemBuilder: (context, index) {
                                                final match =
                                                    upcomingMatches[index];
                                                return AnnouncedMatch(
                                                    winner: match.winner.isEmpty
                                                        ? null
                                                        : match.winner,
                                                    teamAFlag: match.team1Image,
                                                    teamBFlag: match.team2Image,
                                                    matchDate:
                                                        "${match.dateTimeGMT.toLocal().day} ${_getMonthName(match.dateTimeGMT.toLocal().month)} ${match.dateTimeGMT.toLocal().year}",
                                                    matchFormat:
                                                        match.matchType,
                                                    teamA: match.team1,
                                                    teamB: match.team2,
                                                    matchStadium:
                                                        match.series.toString(),
                                                    onTap: (team1, team2, date,
                                                        format) {
                                                      onUpcomingMatchesCardClick(
                                                          team1,
                                                          team2,
                                                          date,
                                                          format);
                                                    },
                                                    matchTime: formatTime(
                                                        match.dateTimeGMT.hour,
                                                        match.dateTimeGMT
                                                            .minute));
                                              },
                                            ),
                                          ),
                                        ),
                                      ],
                                    ],
                                  ),
                      ),

                      // second tab bar view widget
                      Container(
                          alignment: Alignment.topCenter,
                          child: Column(
                            children: [
                              Container(
                                width: double.infinity,
                                padding:
                                    const EdgeInsets.only(left: 12, top: 12),
                                child: const Text(
                                  "Select Match Details",
                                  style: TextStyle(
                                      fontWeight: FontWeight.bold,
                                      fontSize: 22),
                                ),
                              ),
                              TeamSelectionWidget(
                                  onFormatSelect: (v) {
                                    setState(() {
                                      selectedFormat = v;
                                      teamASelectedPlayers = [];
                                      teamBSelectedPlayers = [];
                                      teamAName = '';
                                      teamBName = '';
                                      matchId = '';
                                    });
                                    updateMatchDetailsSelection();
                                  },
                                  onDateSelect: (value) {
                                    setState(() {
                                      selectedDate = value;
                                      teamASelectedPlayers = [];
                                      teamBSelectedPlayers = [];
                                      teamAName = '';
                                      teamBName = '';
                                      matchId = '';
                                    });
                                    updateMatchDetailsSelection();
                                  },
                                  format: selectedFormat.isEmpty
                                      ? null
                                      : selectedFormat,
                                  date: selectedDate.isEmpty
                                      ? null
                                      : selectedDate,
                                  teamA: teamAName,
                                  teamB: teamBName,
                                  onTeamAChange: (value) {
                                    setState(() {
                                      teamAName = '';
                                      matchId = '';
                                      teamASelectedPlayers = [];
                                      teamBSelectedPlayers = [];
                                    });
                                  },
                                  onTeamBChange: (value) {
                                    setState(() {
                                      teamBName = '';
                                      matchId = '';
                                      teamASelectedPlayers = [];
                                      teamBSelectedPlayers = [];
                                    });
                                  },
                                  onNextPressed: (teamA, teamB) {
                                    setState(() {
                                      teamAName = teamA;
                                      teamBName = teamB;
                                      teamASelectedPlayers = [];
                                      teamBSelectedPlayers = [];
                                    });
                                    updateTeam(teamA, teamB);
                                  }),
                              (teamAName != '' &&
                                      teamBName != '' &&
                                      teamAPlayers.isNotEmpty &&
                                      teamAPlayers.isNotEmpty)
                                  ? SelectPlayersWidget(
                                      teamA: TeamModel(
                                        teamName: teamAName,
                                        players: teamAPlayers,
                                      ),
                                      selectedTeamAPlayers:
                                          teamASelectedPlayers,
                                      onTeamAPlayersSelect: (players) {
                                        setState(() {
                                          teamASelectedPlayers = players;
                                        });
                                      },
                                      teamB: TeamModel(
                                          teamName: teamBName,
                                          players: teamBPlayers),
                                      selectedTeamBPlayers:
                                          teamBSelectedPlayers,
                                      onTeamBPlayersSelect: (players) {
                                        setState(() {
                                          teamBSelectedPlayers = players;
                                        });
                                      },
                                    )
                                  : const SizedBox(),
                              const SizedBox(
                                height: 20,
                              ),
                              LargeButton(
                                isLoading: submitButtonClicked,
                                title: "Submit",
                                isDisable: teamASelectedPlayers.length != 11 ||
                                    teamBSelectedPlayers.length != 11,
                                onTap: () async {
                                  setState(() {
                                    submitButtonClicked = true;
                                  });
                                  DateTime parsedDate = DateFormat('d/M/yyyy')
                                      .parse(selectedDate);

                                  String formattedDate =
                                      DateFormat('yyyy-MM-dd')
                                          .format(parsedDate);
                                  showDialog(
                                    barrierColor:
                                        const Color.fromARGB(213, 0, 0, 0),
                                    context: context,
                                    barrierDismissible: false,
                                    builder: (BuildContext context) {
                                      return Dialog(
                                        backgroundColor: Colors.transparent,
                                        child: Container(
                                          decoration: BoxDecoration(
                                            borderRadius:
                                                BorderRadius.circular(10),
                                          ),
                                          child: Column(
                                            mainAxisSize: MainAxisSize.min,
                                            children: [
                                              ClipRRect(
                                                borderRadius:
                                                    BorderRadius.circular(15),
                                                child: Image.asset(
                                                  'assets/images/cric_loading.gif', // Path to your GIF asset
                                                  width: screenWidth,
                                                ),
                                              ),
                                              SizedBox(
                                                height: 20,
                                              ),
                                              LoadingAnimationWidget.waveDots(
                                                color: Colors.white,
                                                size: 50,
                                              ),
                                            ],
                                          ),
                                        ),
                                      );
                                    },
                                  );

                                  List<PlayerPredictionModel> predictedPlayers =
                                      await ApiServices().getPredictionData(
                                          formattedDate,
                                          selectedFormat,
                                          teamASelectedPlayers +
                                              teamBSelectedPlayers,
                                          matchId);
                                  setState(() {
                                    submitButtonClicked = false;
                                  });
                                  Navigator.pop(context);

                                  Navigator.push(
                                    context,
                                    MaterialPageRoute(
                                      builder: (context) => AiTeamScreen(
                                        playerPredictions: predictedPlayers,
                                        teamA: TeamModel(
                                          teamName: teamAName,
                                          players: teamAPlayers,
                                        ),
                                        teamB: TeamModel(
                                          teamName: teamBName,
                                          players: teamBPlayers,
                                        ),
                                      ),
                                    ),
                                  );
                                },
                              ),
                            ],
                          ))
                    ],
                  ),
                ),
              ],
            ),
    );
  }

  String formatTime(int hour, int minute) {
    String period = hour >= 12 ? "PM" : "AM";

    int hourIn12HrFormat = hour % 12;
    hourIn12HrFormat = (hourIn12HrFormat == 0) ? 12 : hourIn12HrFormat;

    String formattedMinute = minute < 10 ? "0$minute" : "$minute";

    return "$hourIn12HrFormat:$formattedMinute $period";
  }
}
