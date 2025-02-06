import 'dart:convert';

import 'package:cricgenius/model/announcedMatch.dart';
import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/model/player_prediction.dart';
import 'package:cricgenius/model/team.dart';
import 'package:dio/dio.dart';
import 'package:flutter/material.dart';

class ApiServices {
  static final ApiServices _instance = ApiServices._internal();
  factory ApiServices() => _instance;

  ApiServices._internal();

  String backendUrl = "http://localhost:8000/";

  Dio dio = Dio();

  Future<void> configureDio() async {
    dio.options.baseUrl = backendUrl;
  }

  Future<List<AnnouncedMatchModel>> getAnnouncedMatches(
      String date, String format) async {
    try {
      final response = await dio.get('/matches/$date/$format');
      if (response.statusCode == 200) {
        var data = response.data;
        List<dynamic> matches;

        if (data is List) {
          matches = data;
        } else if (data is Map) {
          matches = [data];
        } else {
          debugPrint("Unexpected response data type: ${data.runtimeType}");
          return [];
        }

        return matches
            .map((match) => AnnouncedMatchModel.fromJson(match))
            .toList();
      } else {
        debugPrint(
            "Failed to get announced matches with status code ${response.statusCode}");
        return [];
      }
    } on DioException catch (e) {
      debugPrint("Failed to get announced matches, Dio error: ${e.message}");
      return [];
    } catch (e) {
      debugPrint("Failed to get announced matches: $e");
      return [];
    }
  }

  Future<Map<String, String>?> getPlayerImages(
      Map<String, String> playersData) async {
    try {
      final response = await dio.post('/images', data: jsonEncode(playersData));

      if (response.statusCode == 200) {
        Map<String, String> images = Map<String, String>.from(response.data);
        return images;
      } else {
        debugPrint(
            "failed to get players images with status code ${response.statusCode}");
        return null;
      }
    } on DioException catch (e) {
      debugPrint("Dio error: ${e.message}");
      return null;
    } catch (e) {
      debugPrint("Unexpected error: $e");
      return null;
    }
  }

  Future<List<PlayerModel>> searchPlayers(String query) async {
    try {
      final response =
          await dio.get('/players/search', queryParameters: {'q': query});

      if (response.statusCode == 200) {
        final data = response.data;

        if (data != null && data['matching_players'] != null) {
          return (data['matching_players'] as List)
              .map((player) => PlayerModel.fromJson(player))
              .toList();
        }
      }
      return [];
    } on DioException catch (e) {
      debugPrint("Dio error: ${e.message}");
      return [];
    } catch (e) {
      debugPrint("Unexpected error: $e");
      return [];
    }
  }

  Future<List<String>> searchTeams(String query) async {
    try {
      final response =
          await dio.get('/teams/search', queryParameters: {'q': query});

      if (response.statusCode == 200) {
        final data = response.data;

        if (data != null && data['matching_teams'] != null) {
          return (data['matching_teams'] as List)
              .map((team) => team['Team'] as String)
              .toList();
        }
      }

      return [];
    } on DioException catch (e) {
      debugPrint("Dio error: ${e.message}");
      return [];
    } catch (e) {
      debugPrint("Unexpected error: $e");
      return [];
    }
  }

  Future<TeamModel?> getTeamData(String teamName) async {
    try {
      final response = await dio.get('/team/$teamName');

      if (response.statusCode == 200) {
        final data = response.data;

        if (data != null) {
          return TeamModel.fromJson(data);
        }
      } else {
        debugPrint(
            "Failed to fetch team data. Status code: ${response.statusCode}");
      }
      return null;
    } catch (e) {
      debugPrint("Error fetching team data: $e");
      return null;
    }
  }

  Future<List<PlayerPredictionModel>> getPredictionData(
      String date, format, List<PlayerModel> players, String matchId) async {
    if (format == "TEST") {
      format = "Test";
    }
    final data = {
      "date": date,
      "format": format == "TEST" ? "Test" : format,
      "players_id_list":
          players.map((player) => player.uniqueIdentifier).toList(),
      "match_id": matchId,
    };

    try {
      final response = await dio.post(
        'api/true_predict',
        data: data,
        options: Options(
          headers: {
            "Content-Type": "application/json",
          },
        ),
      );

      if (response.statusCode == 200) {
        final predictionData = response.data as Map<String, dynamic>;

        // Convert the response data into a list of PlayerPredictionModel
        List<PlayerPredictionModel> predictions = [];
        predictionData.forEach((key, value) {
          predictions.add(PlayerPredictionModel.fromJson(key, value));
        });

        return predictions;
      } else {
        debugPrint("Error: ${response.statusCode} - ${response.data}");
        return [];
      }
    } catch (e) {
      if (e is DioError) {
        debugPrint("DioError: ${e.response?.data ?? e.message}");
        return [];
      } else {
        debugPrint("Failed to send data: $e");
        return [];
      }
    }
  }

  Future<dynamic> getUpcomingMatches() async {
    try {
      final response = await dio.get('/matches/upcoming');

      if (response.statusCode == 200) {
        final data = response.data;
        return data;
      } else {}
      return null;
    } catch (e) {
      return null;
    }
  }

  Future<PlayerModel?> getPlayerInfo(String identifier) async {
    try {
      final response = await dio.get('/api/player/$identifier');

      if (response.statusCode == 200) {
        final data = response.data;
        return PlayerModel.fromInfoJson(data);
      } else {
        return null;
      }
    } catch (e) {
      return null;
    }
  }
}
