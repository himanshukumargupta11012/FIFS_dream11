import 'package:cricgenius/model/player.dart';
import 'package:flutter/material.dart';

class PlayerSelectionWidget extends StatefulWidget {
  const PlayerSelectionWidget({
    super.key,
    required this.teamA,
    required this.teamB,
    required this.teamAPlayers,
    required this.teamBPlayers,
    required this.onChange,
  });

  final String teamA; // Name of Team A
  final String teamB; // Name of Team B
  final List<PlayerModel> teamAPlayers; // List of players in Team A
  final List<PlayerModel> teamBPlayers; // List of players in Team B
  final Function(String name, bool isSelected, int team)
      onChange; // Callback when selection changes

  @override
  // ignore: library_private_types_in_public_api
  _PlayerSelectionWidgetState createState() => _PlayerSelectionWidgetState();
}

class _PlayerSelectionWidgetState extends State<PlayerSelectionWidget> {
  final Set<String> selectedPlayersTeamA = {};
  final Set<String> selectedPlayersTeamB = {};

  // Function to show an error messages using a SnackBar
  void showError({String? msg}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Center(child: Text(msg ?? 'Something went wrong!')),
        duration: const Duration(milliseconds: 1500),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  // display a player card that can be selected or deselected
  Widget card(String name, bool isSelected, int team) {
    return GestureDetector(
      onTap: () {
        setState(() {
          if (isSelected) {
            // If already selected, remove the player from the selection
            if (team == 0) {
              selectedPlayersTeamA.remove(name);
            } else {
              selectedPlayersTeamB.remove(name);
            }
          } else {
            // If not selected, add the player to the selection if limit not reached
            if (team == 0) {
              if (selectedPlayersTeamA.length < 11) {
                selectedPlayersTeamA.add(name);
              } else {
                showError(
                    msg:
                        "You can only select up to 11 players for ${widget.teamA}.");
                return;
              }
            } else {
              if (selectedPlayersTeamB.length < 11) {
                selectedPlayersTeamB.add(name);
              } else {
                showError(
                    msg:
                        "You can only select up to 11 players for ${widget.teamB}");
                return;
              }
            }
          }
          // Notify about the change
          widget.onChange(name, isSelected, team);
        });
      },
      child: Container(
        margin: const EdgeInsets.symmetric(vertical: 4),
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(12),
          color: isSelected
              ? const Color(0xffA3ECA6)
              : const Color(0xff020D07).withOpacity(0.12),
        ),
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
          child: Row(
            children: [
              Expanded(
                child: Text(
                  name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Colors.black,
                  ),
                ),
              ),
              const SizedBox(width: 8),
              Icon(
                isSelected ? Icons.remove : Icons.add,
                size: 18,
                color: Colors.black,
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Column(
        children: [
          // Display team names at the top
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              Expanded(
                child: Center(
                  child: Text(
                    widget.teamA,
                    style: const TextStyle(
                        fontSize: 22, fontWeight: FontWeight.w800),
                  ),
                ),
              ),
              Expanded(
                child: Center(
                  child: Text(
                    widget.teamB,
                    style: const TextStyle(
                        fontSize: 22, fontWeight: FontWeight.w800),
                  ),
                ),
              ),
            ],
          ),
          const Divider(thickness: 1, color: Colors.black),
          // Display lists of players from both teams
          Expanded(
            child: Row(
              children: [
                // Team A players
                Expanded(
                  child: ListView.builder(
                    itemCount: widget.teamAPlayers.length,
                    itemBuilder: (context, index) {
                      final playerName = widget.teamAPlayers[index];
                      return card(
                        playerName.name,
                        selectedPlayersTeamA.contains(playerName),
                        0,
                      );
                    },
                  ),
                ),
                const SizedBox(width: 12),
                // Team B players
                Expanded(
                  child: ListView.builder(
                    itemCount: widget.teamBPlayers.length,
                    itemBuilder: (context, index) {
                      final playerName = widget.teamBPlayers[index];
                      return card(
                        playerName.name,
                        selectedPlayersTeamB.contains(playerName),
                        1,
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
