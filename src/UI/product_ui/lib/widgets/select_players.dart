import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/model/team.dart';
import 'package:cricgenius/widgets/player_add_bottom_sheet.dart';
import 'package:flutter/material.dart';

class SelectPlayersWidget extends StatelessWidget {
  const SelectPlayersWidget({
    super.key,
    required this.teamA,
    required this.teamB,
    required this.selectedTeamAPlayers,
    required this.selectedTeamBPlayers,
    required this.onTeamAPlayersSelect,
    required this.onTeamBPlayersSelect,
  });

  final TeamModel teamA, teamB;
  final List<PlayerModel> selectedTeamAPlayers, selectedTeamBPlayers;
  final Function(List<PlayerModel> player) onTeamAPlayersSelect,
      onTeamBPlayersSelect;

  Widget oneTeamRow(
      BuildContext context,
      TeamModel team,
      int selectedPlayersCount,
      List<PlayerModel> selectedPlayers,
      Function(List<PlayerModel> player) onPlayersSelect) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Expanded(
          child: Row(
            children: [
              Image.network(
                team.getTeamLogo(),
                width: 24,
              ),
              const SizedBox(
                width: 4,
              ),
              Expanded(
                child: Text(
                  team.teamName,
                  overflow: TextOverflow
                      .ellipsis, // Adds ellipsis when text overflows
                  style: const TextStyle(
                    color: Colors.black87,
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),
              const SizedBox(
                width: 4,
              ),
              Container(
                width: 24,
                height: 24,
                decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(24),
                    color: selectedPlayersCount == 11
                        ? const Color(0xff40B18A)
                        : const Color(0xffF64646)),
                child: Center(
                    child: Text(
                  "$selectedPlayersCount",
                  style: const TextStyle(color: Colors.white),
                )),
              )
            ],
          ),
        ),
        const SizedBox(
          width: 5,
        ),
        InkWell(
          onTap: () {
            showModalBottomSheet(
              isScrollControlled: true,
              context: context,
              isDismissible: true,
              builder: (context) => PlayerAddBottomSheet(
                image:
                    team.getTeamLogo(),
                name: team.teamName,
                players: team.players,
                selectedPlayers: selectedPlayers,
                onPlayersSelect: onPlayersSelect,
              ),
            );
          },
          child: Container(
            padding: const EdgeInsets.fromLTRB(18, 4, 18, 4),
            decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                color: const Color(0xff0A6DEA)),
            child: const Text(
              "Select Players",
              style: TextStyle(color: Colors.white, fontSize: 16),
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12),
      width: double.infinity,
      height: 97,
      decoration: ShapeDecoration(
        color: Colors.white,
        shape: RoundedRectangleBorder(
          side: const BorderSide(width: 0.20, color: Color(0xFF8E8E8E)),
          borderRadius: BorderRadius.circular(21),
        ),
        shadows: const [
          BoxShadow(
            color: Color(0x3F000000),
            blurRadius: 9.80,
            offset: Offset(0, 6),
            spreadRadius: -9,
          )
        ],
      ),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12),
        child: Column(
          children: [
            const SizedBox(
              height: 12,
            ),
            oneTeamRow(context, teamA, selectedTeamAPlayers.length,
                selectedTeamAPlayers, onTeamAPlayersSelect),
            const SizedBox(
              height: 8,
            ),
            oneTeamRow(context, teamB, selectedTeamBPlayers.length,
                selectedTeamBPlayers, onTeamBPlayersSelect),
            // SizedBox(
            //   height: 8,
            // ),
          ],
        ),
      ),
    );
  }
}
