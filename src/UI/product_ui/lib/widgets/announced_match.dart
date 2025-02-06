import 'package:flutter/material.dart';

// Card to be shown if there are any upcoming matches scheduled
class AnnouncedMatch extends StatelessWidget {
  final String teamA;
  final String teamB;
  String? teamAFlag;
  String? teamBFlag;

  // circket formats like ODI, Test, T20
  final String matchFormat;
  final String matchStadium;
  final String matchDate;

  /// matchTime is reduntant as of now
  final String? matchTime;

  final String? winner;

  // callback function
  final Function(String teamA, String teamB, String date, String format)? onTap;

  AnnouncedMatch(
      {super.key,
      required this.teamA,
      required this.teamB,
      this.teamAFlag,
      this.teamBFlag,
      required this.matchFormat,
      required this.matchStadium,
      required this.matchDate,
      this.matchTime,
      this.winner,
      this.onTap});

  Widget pastMatchWidget() {
    return Container(
      width: 240,
      height: 120,

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
      // color: Colors.white,
      margin: const EdgeInsets.fromLTRB(10.0, 0, 10, 10),
      child: InkWell(
        borderRadius: BorderRadius.circular(12),
        onTap: () {
          if (onTap != null) {
            onTap!(teamA, teamB, matchDate, matchFormat);
          }
        },
        child: Padding(
          padding: const EdgeInsets.all(12.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '$matchDate • ${matchFormat.toUpperCase()}',
                style: const TextStyle(
                  fontWeight: FontWeight.bold,
                  color: Colors.black,
                  fontSize: 15,
                ),
              ),
              const SizedBox(height: 10.0),
              Column(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      teamAFlag != null
                          ? Image.network(
                              teamAFlag!,
                              width: 20,
                              height: 12,
                              fit: BoxFit.cover,
                            )
                          : const SizedBox(),
                      const SizedBox(
                        width: 5,
                      ),
                      Text(
                        teamA,
                        style: const TextStyle(
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    mainAxisAlignment: MainAxisAlignment.start,
                    children: [
                      teamBFlag != null
                          ? Image.network(
                              teamBFlag!,
                              width: 20,
                              height: 12,
                              fit: BoxFit.cover,
                            )
                          : const SizedBox(),
                      const SizedBox(
                        width: 5,
                      ),
                      Text(
                        teamB,
                        style: const TextStyle(
                          fontSize: 12,
                        ),
                      ),
                    ],
                  ),
                ],
              ),
              const SizedBox(
                height: 4,
              ),
              if (winner != null) ...[
                const SizedBox(
                  height: 4,
                ),
                Text(
                  winner ?? '',
                  style: const TextStyle(
                      color: Color.fromARGB(255, 85, 85, 85),
                      fontSize: 11,
                      fontStyle: FontStyle.italic),
                ),
              ]
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return winner != null
        ? pastMatchWidget()
        : Container(
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
            // color: Colors.white,
            margin: const EdgeInsets.fromLTRB(10.0, 0, 10, 10),
            child: InkWell(
              borderRadius: BorderRadius.circular(12),
              onTap: () {
                if (onTap != null) {
                  onTap!(teamA, teamB, matchDate, matchFormat);
                }
              },
              child: Padding(
                padding: const EdgeInsets.all(12.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Text(
                          '$matchDate • ${matchFormat.toUpperCase()}',
                          style: const TextStyle(
                            fontWeight: FontWeight.bold,
                            color: Colors.black,
                            fontSize: 18,
                          ),
                        ),
                        if (matchTime != null)
                          Text(
                            '$matchTime',
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              color: Colors.black,
                              fontSize: 12,
                            ),
                          ),
                      ],
                    ),
                    const SizedBox(height: 10.0),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Expanded(
                          child: Row(
                            children: [
                              teamAFlag != null
                                  ? Image.network(
                                      teamAFlag!,
                                      width: 25,
                                      height: 15,
                                      fit: BoxFit.cover,
                                    )
                                  : const SizedBox(),
                              const SizedBox(
                                width: 5,
                              ),
                              Flexible(
                                child: Text(
                                  teamA,
                                  overflow: TextOverflow.ellipsis,
                                  style: const TextStyle(
                                    fontSize: 16,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        Container(
                          margin: EdgeInsets.symmetric(horizontal: 6),
                          child: const Text(
                            'vs',
                            style: TextStyle(
                              color: Color.fromARGB(255, 85, 85, 85),
                              fontSize: 14,
                            ),
                          ),
                        ),
                        Expanded(
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.end,
                            children: [
                              teamBFlag != null
                                  ? Image.network(
                                      teamBFlag!,
                                      width: 25,
                                      height: 15,
                                      fit: BoxFit.cover,
                                    )
                                  : const SizedBox(),
                              const SizedBox(
                                width: 5,
                              ),
                              Flexible(
                                child: Text(
                                  teamB,
                                  overflow: TextOverflow.ellipsis,
                                  style: const TextStyle(
                                    fontSize: 16,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(
                      height: 4,
                    ),
                    Text(
                      matchStadium,
                      style: const TextStyle(
                        color: Color.fromARGB(255, 85, 85, 85),
                        fontSize: 14,
                      ),
                    ),
                    if (winner != null) ...[
                      const SizedBox(
                        height: 4,
                      ),
                      Text(
                        winner ?? '',
                        style: const TextStyle(
                            color: Color.fromARGB(255, 85, 85, 85),
                            fontSize: 14,
                            fontStyle: FontStyle.italic),
                      ),
                    ]
                  ],
                ),
              ),
            ),
          );
  }
}
