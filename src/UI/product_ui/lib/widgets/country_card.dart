import 'package:flutter/material.dart';

class CountryCard extends StatelessWidget {
  final String countryName;
  final String flagUrl;
  final bool isLeft;

  const CountryCard(
      {super.key, required this.countryName, required this.flagUrl, required this.isLeft});

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      color: const Color(0xFFFFC850),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.only(
            topLeft: Radius.circular(!isLeft ? 20 : 0),
            bottomLeft: Radius.circular(!isLeft ? 20 : 0),
            topRight: Radius.circular(isLeft ? 20 : 0),
            bottomRight: Radius.circular(isLeft ? 20 : 0),
            ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(8.0),
        child: Row(
          children: [
            Image.network(
              flagUrl,
              width: 25,
              height: 15,
              fit: BoxFit.cover,
            ),
            const SizedBox(width: 5),
            Text(
              countryName,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
