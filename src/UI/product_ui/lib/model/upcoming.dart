import 'package:flutter/services.dart';

class Match {
  final String id;
  final DateTime dateTimeGMT;
  final String matchType;
  final String status;
  final String ms;
  final String team1;
  final String team2;
  final String team1Score;
  final String team2Score;
  final String team1Image;
  final String team2Image;
  final String series;
  final String winner;
  final bool isUpcoming;

  Match({
    required this.id,
    required this.dateTimeGMT,
    required this.matchType,
    required this.status,
    required this.ms,
    required this.team1,
    required this.team2,
    required this.team1Score,
    required this.team2Score,
    required this.team1Image,
    required this.team2Image,
    required this.series,
    required this.winner,
    required this.isUpcoming,
  });

  factory Match.fromJson(Map<String, dynamic> json) {
    String winner = extractWinner(json['status'] ?? '', json['ms'] ?? '');

    bool isMatchStarted =
        statusContainsMatchStarted(json['status'] ?? '', json['ms'] ?? '');

    // According to the requirement:
    // If the match has started or ended, it's a past match (isUpcoming = false)
    // Otherwise, it's an upcoming match (isUpcoming = true)
    bool isUpcoming = !isMatchStarted;

    String rawTeam1 = json['t1'] ?? 'Unknown Team 1';
    String rawTeam2 = json['t2'] ?? 'Unknown Team 2';

    String cleanTeam1 = removeBracketedText(rawTeam1);
    String cleanTeam2 = removeBracketedText(rawTeam2);

    String gmtTime = json['dateTimeGMT'] ?? DateTime.now().toString();

    return Match(
      id: json['id'] ?? '',
      dateTimeGMT: DateTime.parse(gmtTime),
      matchType: json['matchType'] ?? '',
      status: json['status'] ?? '',
      ms: json['ms'] ?? '',
      team1: cleanTeam1,
      team2: cleanTeam2,
      team1Score: json['t1s'] ?? '',
      team2Score: json['t2s'] ?? '',
      team1Image: enhanceImage((json['t1img']) ??
          "https://iiwiki.us/mediawiki/images/thumb/5/57/Placeholder_Flag.png/300px-Placeholder_Flag.png"),
      team2Image: enhanceImage(json['t2img'] ??
          "https://iiwiki.us/mediawiki/images/thumb/5/57/Placeholder_Flag.png/300px-Placeholder_Flag.png"),
      series: json['series'] ?? '',
      winner: winner,
      isUpcoming: isUpcoming,
    );
  }

  static String removeBracketedText(String input) {
    return input.replaceAll(RegExp(r'\s*\[.*?\]'), '').trim();
  }

  static String enhanceImage(String url) {
    return '${url.split("=")[0]}=48';
  }

  /// Extracts the winner's name from the status string.
  /// If the match is not started or is a fixture, returns an empty string.
  static String extractWinner(String status, String ms) {
    if (ms.toLowerCase() == "result") {
      return status;
    } else {
      return "";
    }
  }

  /// Determines if the match has started based on the status and ms fields.
  static bool statusContainsMatchStarted(String status, String ms) {
    if (ms.toLowerCase() == "result") {
      return true;
    }
    return false;
  }
}

class ApiResponse {
  final List<Match> matches;
  final String status;
  // final Info info;

  ApiResponse({
    required this.matches,
    required this.status,
    // required this.info,
  });

  factory ApiResponse.fromJson(
      Map<String, dynamic> json, Map<String, dynamic> teamsData) {
    List<String> validTeams = (teamsData['Teams'] as List<dynamic>)
        .map((e) => e.toString().toLowerCase())
        .toList();

    var matchesJson = json['data'] as List<dynamic>? ?? [];
    var filteredMatchesJson = matchesJson.where((matchJson) {
      String team1 = (matchJson['t1'] ?? '')
          .toString()
          .toLowerCase()
          .replaceAll(RegExp(r'\s*\[.*?\]'), '')
          .trim();
      String team2 = (matchJson['t2'] ?? '')
          .toString()
          .toLowerCase()
          .replaceAll(RegExp(r'\s*\[.*?\]'), '')
          .trim();

      return validTeams.contains(team1) &&
          validTeams.contains(team2) &&
          !team1.contains('women') &&
          !team2.contains('women');
    }).toList();

    List<Match> matches = filteredMatchesJson
        .map((matchJson) => Match.fromJson(matchJson))
        .toList();

    return ApiResponse(
      matches: matches,
      status: json['status'] ?? '',
      // info: Info.fromJson(json['info'] ?? {}),
    );
  }
}

class Info {
  final int hitsToday;
  final int hitsUsed;
  final int hitsLimit;
  final int credits;
  final int server;
  final double queryTime;
  final int s;

  Info({
    required this.hitsToday,
    required this.hitsUsed,
    required this.hitsLimit,
    required this.credits,
    required this.server,
    required this.queryTime,
    required this.s,
  });

  factory Info.fromJson(Map<String, dynamic> json) {
    return Info(
      hitsToday: json['hitsToday'] ?? 0,
      hitsUsed: json['hitsUsed'] ?? 0,
      hitsLimit: json['hitsLimit'] ?? 0,
      credits: json['credits'] ?? 0,
      server: json['server'] ?? 0,
      queryTime: (json['queryTime'] ?? 0).toDouble(),
      s: json['s'] ?? 0,
    );
  }
}
