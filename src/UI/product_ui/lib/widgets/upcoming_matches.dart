import 'package:flutter/material.dart';

class UpcomingMatchesWidget extends StatelessWidget {
  final String date;
  final String type;
  final String teamA;
  final String teamB;
  final String venue;
  final String teamAFlag;
  final String teamBFlag;

  const UpcomingMatchesWidget(
      {super.key,
      required this.date,
      required this.type,
      required this.teamA,
      required this.teamB,
      required this.venue,
      required this.teamAFlag,
      required this.teamBFlag});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 12),
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 6),
      height: 87,
      decoration: ShapeDecoration(
        color: const Color(0xFFFDFDFD),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(10),
        ),
        shadows: const [
          BoxShadow(
            color: Color(0x3F000000),
            blurRadius: 13.80,
            offset: Offset(0, 2),
            spreadRadius: -4,
          )
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            margin: const EdgeInsets.only(left: 12),
            child: Text(
              "$date • $type",
              style: const TextStyle(
                color: Colors.black,
                fontSize: 18,
                fontWeight: FontWeight.w600,
              ),
            ),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                margin: const EdgeInsets.only(left: 12),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Image.network(
                      teamAFlag,
                      width: 24,
                    ),
                    const SizedBox(
                      width: 4,
                    ),
                    Text(
                      teamA,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.black87,
                        fontSize: 18,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              const Text("vs"),
              Container(
                margin: const EdgeInsets.only(right: 12),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Image.network(
                      teamBFlag,
                      width: 24,
                    ),
                    const SizedBox(
                      width: 4,
                    ),
                    Text(
                      teamB,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(
                        color: Colors.black87,
                        fontSize: 18,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          Container(
            margin: const EdgeInsets.only(left: 12, top: 4),
            child: Text(
              venue,
              style: const TextStyle(
                color: Color(0xff424242),
                fontSize: 12,
              ),
            ),
          )
        ],
      ),
    );
  }
}
