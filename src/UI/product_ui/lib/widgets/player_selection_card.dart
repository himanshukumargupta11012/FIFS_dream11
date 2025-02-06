import 'package:flutter/material.dart';

class PlayerSelectionCard extends StatelessWidget {
  final String name;
  // final String role;
  final String imageUrl;
  // final String stats;
  final bool isSelected;

  const PlayerSelectionCard({
    super.key,
    required this.name,
    // required this.role,
    required this.imageUrl,
    // required this.stats,
    required this.isSelected,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: isSelected ? 8.0 : 2.0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Image.network(imageUrl, height: 140, fit: BoxFit.cover),
            const SizedBox(height: 10),
            Text(
              name,
              style: const TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
                height: 0.9,
              ),
            ),
            const SizedBox(
              height: 5,
            ),
            // Container(
            //   padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 2),
            //   decoration: BoxDecoration(
            //     color: const Color(0xFF203591),
            //     borderRadius: BorderRadius.circular(10),
            //   ),
            //   child: Text(
            //     role,
            //     style: const TextStyle(
            //       color: Colors.white,
            //       fontSize: 14,
            //     ),
            //   ),
            // ),
            const SizedBox(
              height: 10,
            ),
            // Text(
            //   stats,
            //   textAlign: TextAlign.center,
            //   style: const TextStyle(fontSize: 12, color: Colors.black54),
            // ),
          ],
        ),
      ),
    );
  }
}
