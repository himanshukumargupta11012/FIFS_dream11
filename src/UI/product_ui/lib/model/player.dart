class PlayerModel {
  final String name;
  final String uniqueIdentifier;
  final String? battingStyle;
  final String? bowlingStyle;
  final String? playingRole;

  const PlayerModel({
    required this.name,
    required this.uniqueIdentifier,
    this.battingStyle,
    this.bowlingStyle,
    this.playingRole,
  });

  factory PlayerModel.fromJson(Map<String, dynamic> json) {
    return PlayerModel(
      name: json['name'],
      uniqueIdentifier: json['identifier'],
    );
  }

    factory PlayerModel.fromInfoJson(Map<String, dynamic> json) {
    return PlayerModel(
      name: json['full_name'],
      uniqueIdentifier: json['identifier'],
      battingStyle: json['batting_style'],
      bowlingStyle: json['bowling_style'],
      playingRole: json['playing_role'],
    );
  }

  String getPlayerImage() {
    return 'http://localhost:8000/cdn/$uniqueIdentifier';
  }
}
