import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/widgets/large_button.dart';
import 'package:cricgenius/widgets/player_add_card.dart';
import 'package:flutter/material.dart';

class PlayerAddBottomSheet extends StatefulWidget {
  const PlayerAddBottomSheet({
    super.key,
    required this.image,
    required this.name,
    required this.players,
    required this.selectedPlayers,
    required this.onPlayersSelect,
  });

  final String image, name;
  final List<PlayerModel> players, selectedPlayers;
  final Function(List<PlayerModel> player) onPlayersSelect;

  @override
  State<PlayerAddBottomSheet> createState() => _PlayerAddBottomSheetState();
}

class _PlayerAddBottomSheetState extends State<PlayerAddBottomSheet> {
  List<PlayerModel> localSelectedPlayers = [];
  List<PlayerModel> filteredPlayers = [];
  List<PlayerModel> localPlayers = [];
  String playerSearch = '';

  TextEditingController playerSearchController = TextEditingController();
  @override
  void initState() {
    super.initState();
    localSelectedPlayers = widget.selectedPlayers;

    localPlayers = widget.players;

    playerSearchController.addListener(() {
      setState(() {
        String searchQuery = playerSearchController.text.toLowerCase();
        if (searchQuery.isEmpty) {
          localPlayers = widget.players;
        } else {
          localPlayers = widget.players
              .where(
                  (player) => player.name.toLowerCase().contains(searchQuery))
              .toList();
        }
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    double screenWidth = MediaQuery.of(context).size.width;

    return Padding(
      padding:
          EdgeInsets.only(bottom: MediaQuery.of(context).viewInsets.bottom),
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(19),
          boxShadow: const [
            BoxShadow(
              color: Color(0x3F000000),
              blurRadius: 25,
              offset: Offset(0, -6),
              spreadRadius: 5,
            )
          ],
        ),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.start,
          children: [
            const SizedBox(
              height: 12,
            ),
            Container(
              height: 5,
              width: screenWidth / 3,
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(20),
                color: const Color(0xFFD9D9D9),
              ),
            ),
            const SizedBox(
              height: 20,
            ),
            Padding(
              padding:
                  const EdgeInsets.symmetric(vertical: 5.0, horizontal: 10),
              child: Row(
                children: [
                  Image.network(
                    widget.image,
                    height: 35,
                  ),
                  const SizedBox(
                    width: 10,
                  ),
                  Text(
                    widget.name.toUpperCase(),
                    style: const TextStyle(
                        fontSize: 20, fontWeight: FontWeight.w600),
                  ),
                ],
              ),
            ),
            Container(
              alignment: Alignment.centerLeft,
              margin: EdgeInsets.fromLTRB(10, 0, 0, 0),
              child: Text(
                textAlign: TextAlign.start,
                localSelectedPlayers.isNotEmpty
                    ? 'Selected ${localSelectedPlayers.length} Players'
                    : 'Selected no player',
                style: const TextStyle(
                    color: Colors.grey,
                    fontSize: 20,
                    fontWeight: FontWeight.w500),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
              child: Row(
                children: [
                  Expanded(
                    child: TextField(
                      controller: playerSearchController,
                      decoration: InputDecoration(
                        hintText: 'Search Player Name',
                        hintStyle: const TextStyle(
                            color: Color(0xFF0A6DEA), fontSize: 16),
                        enabledBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide:
                              const BorderSide(color: Color(0xFF0A6DEA)),
                        ),
                        focusedBorder: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(12),
                          borderSide: const BorderSide(
                              color: Color(0xFF0A6DEA), width: 2),
                        ),
                        contentPadding: const EdgeInsets.symmetric(
                            vertical: 0.0, horizontal: 12.0),
                      ),
                    ),
                  ),
                  const SizedBox(
                    width: 10,
                  ),
                  InkWell(
                    onTap: () {
                      setState(() {
                        playerSearchController.text = '';
                        filteredPlayers = [];
                      });
                    },
                    child: Container(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 18, vertical: 12),
                      decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(12),
                          color: const Color(0xff0A6DEA)),
                      child: const Text(
                        "Clear",
                        style: TextStyle(color: Colors.white, fontSize: 18),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Container(
              constraints: const BoxConstraints(
                maxHeight: 200,
              ),
              child: SingleChildScrollView(
                child: Wrap(
                  spacing: 10,
                  runSpacing: 10,
                  children: [
                    ...localPlayers.map<Widget>((player) {
                      bool isAdded = localSelectedPlayers.contains(player);

                      return InkWell(
                        onTap: () {
                          setState(() {
                            if (isAdded) {
                              localSelectedPlayers.remove(player);
                            } else {
                              if (localSelectedPlayers.length >= 11) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(
                                    content: Text(
                                        "You cannot select more than 11 players."),
                                    backgroundColor: Colors.red,
                                    behavior: SnackBarBehavior.floating,
                                    duration: Duration(seconds: 2),
                                  ),
                                );
                                return;
                              }
                              localSelectedPlayers.add(player);
                            }
                            isAdded = !isAdded;
                          });

                          widget.onPlayersSelect(localSelectedPlayers);
                          (context as Element).markNeedsBuild();
                        },
                        child: SizedBox(
                          width: MediaQuery.of(context).size.width / 2 - 10,
                          child: PlayerAddCard(
                            image: player.getPlayerImage(),
                            name: player.name,
                            isAdded: localSelectedPlayers.contains(player),
                          ),
                        ),
                      );
                    }),
                  ],
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 10),
              child: LargeButton(
                title: "Done",
                onTap: () {
                  Navigator.pop(context);
                },
              ),
            ),
          ],
        ),
      ),
    );
  }
}
