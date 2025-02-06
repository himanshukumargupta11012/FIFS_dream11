import 'package:cricgenius/model/player.dart';
import 'package:intl/intl.dart';

class AnnouncedMatchModel {
  final String team1;
  final String team2;
  final String matchFormat;
  final String matchStadium;
  final String matchTime;
  final String matchDate;
  final String matchId;
  final List<PlayerModel> teamAPlayers;
  final List<PlayerModel> teamBPlayers;

  const AnnouncedMatchModel(
      {required this.team1,
      required this.team2,
      required this.matchFormat,
      required this.matchStadium,
      required this.teamAPlayers,
      required this.teamBPlayers,
      required this.matchDate,
      required this.matchId,
      this.matchTime = ''});

  factory AnnouncedMatchModel.fromJson(Map<String, dynamic> json) {
    final registry =
        Map<String, String>.from(json['registry'] as Map<String, dynamic>);
    final playersJson = json['players'] as Map<String, dynamic>;
    final teamA = json['teamA'] as String;
    final teamB = json['teamB'] as String;
    final matchId = json['match_id'] as String;
    final dateStr = json['date'] as String;

    DateTime dateTime = DateTime.parse(dateStr);
    // final String date = DateFormat('dd MMM yyyy', 'fr').format(dateTime);
    final String date = DateFormat('dd MMM yyyy', 'en').format(dateTime);

    final teamAPlayersNames =
        List<String>.from(playersJson[teamA] as List<dynamic>);
    final teamBPlayersNames =
        List<String>.from(playersJson[teamB] as List<dynamic>);

    final teamAPlayers = teamAPlayersNames.map((name) {
      return PlayerModel(
        name: name,
        uniqueIdentifier: registry[name] ?? '',
      );
    }).toList();

    final teamBPlayers = teamBPlayersNames.map((name) {
      return PlayerModel(
        name: name,
        uniqueIdentifier: registry[name] ?? '',
      );
    }).toList();

    return AnnouncedMatchModel(
        team1: teamA,
        team2: teamB,
        matchFormat: json['format'] as String,
        matchStadium: json['venue'] as String,
        teamAPlayers: teamAPlayers,
        teamBPlayers: teamBPlayers,
        matchId: matchId,
        matchDate: date);
  }
}
