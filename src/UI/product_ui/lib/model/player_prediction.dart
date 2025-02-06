class PlayerPredictionModel {
  final String uniqueIdentifier;
  final String name;
  final String explain;
  final int score;

  const PlayerPredictionModel(
      {required this.uniqueIdentifier,
      required this.name,
      required this.explain,
      required this.score});

  factory PlayerPredictionModel.fromJson(
      String uniqueIdentifier, Map<String, dynamic> json) {
    return PlayerPredictionModel(
      uniqueIdentifier: uniqueIdentifier,
      name: json['name'],
      explain: json['explanation'],
      score: json['score'],
    );
  }

  String getPlayerImage() {
    return 'http://localhost:8000/cdn/$uniqueIdentifier';
  }
}
