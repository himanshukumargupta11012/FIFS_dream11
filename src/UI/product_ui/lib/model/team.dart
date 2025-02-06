import 'package:cricgenius/model/player.dart';

class TeamModel {
  final String teamName;
  final List<PlayerModel> players;

  const TeamModel({
    required this.teamName,
    required this.players,
  });

  factory TeamModel.fromJson(Map<String, dynamic> json) {
    return TeamModel(
      teamName: json['team_name'],
      players: (json['players'] as List)
          .map((player) => PlayerModel.fromJson(player))
          .toList(),
    );
  }

  String getTeamLogo() {
    const baseUrl = "http://localhost:8000/logo/";
    return "$baseUrl${Uri.encodeComponent(teamName)}";
  }

  String getAbbreviation() {
    List<String> words = teamName.split(' ');

    if (words.length > 1) {
      return words.map((word) => word[0]).join('');
    } else {
      return words[0].substring(0, 3);
    }
  }
}
