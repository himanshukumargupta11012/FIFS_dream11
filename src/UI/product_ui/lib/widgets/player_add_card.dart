import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';

class PlayerAddCard extends StatelessWidget {
  const PlayerAddCard({
    super.key,
    required this.image,
    required this.name,
    required this.isAdded,
  });

  final String image;
  final String name;
  final bool isAdded;

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 5, horizontal: 5),
      decoration: BoxDecoration(
          border: Border.all(
              color: isAdded
                  ? const Color.fromARGB(69, 10, 109, 229)
                  : Colors.grey,
              width: isAdded ? 4 : 0.5),
          borderRadius: BorderRadius.circular(10)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          CircleAvatar(
            radius: 20,
            backgroundImage: NetworkImage(image),
          ),
          const SizedBox(
            width: 5,
          ),
          Flexible(
            child: Text(
              name,
              style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500, height: 0.9),
              softWrap: true,
              overflow: TextOverflow.clip,
              maxLines: 2,
              
            ),
          ),
          const SizedBox(
            width: 5,
          ),
          isAdded
              ? const Icon(
                  CupertinoIcons.add_circled,
                  color: Colors.green,
                )
              : const Icon(
                  CupertinoIcons.clear_circled,
                  color: Colors.red,
                ),
        ],
      ),
    );
  }
}
