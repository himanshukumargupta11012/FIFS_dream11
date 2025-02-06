import 'package:flutter/material.dart';
import 'package:cricgenius/widgets/team_autocomplete.dart';
import 'package:intl/intl.dart';

class TeamSelectionWidget extends StatefulWidget {
  const TeamSelectionWidget({
    super.key,
    this.onTeam1Select,
    this.onTeam2Select,
    required this.onFormatSelect,
    required this.onDateSelect,
    required this.onTeamAChange,
    required this.onTeamBChange,
    required this.onNextPressed,
    this.date,
    this.format,
    this.teamA,
    this.teamB,
  });

  final Function(String value)? onTeam1Select;
  final Function(String value)? onTeam2Select;
  final Function(String value) onFormatSelect;
  final Function(String value) onDateSelect;
  final Function(String value) onTeamAChange;
  final Function(String value) onTeamBChange;
  final Function(String teamA, String teamB) onNextPressed;

  final String? date;
  final String? format;
  final String? teamA;
  final String? teamB;

  @override
  _TeamSelectionWidgetState createState() => _TeamSelectionWidgetState();
}

class _TeamSelectionWidgetState extends State<TeamSelectionWidget> {
  String? teamA;
  String? teamB;
  String? selectedFormat;
  String? selectedDate;

  List<String> formats = ["T20", "ODI", "TEST"];

  void _selectDate(BuildContext context, DateTime date) async {
    DateTime? pickedDate = await showDatePicker(
      context: context,
      initialDate: date,
      firstDate: DateTime(2000),
      lastDate: DateTime(2100),
    );
    if (pickedDate != null) {
      setState(() {
        selectedDate =
            "${pickedDate.day}/${pickedDate.month}/${pickedDate.year}";
      });
      widget.onDateSelect(
          "${pickedDate.day}/${pickedDate.month}/${pickedDate.year}" ?? '');
    }
  }

  @override
  void initState() {
    super.initState();
    selectedDate = widget.date;
    selectedFormat = widget.format;
    teamA = widget.teamA;
    teamB = widget.teamB;
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16.0),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 12.0),
        decoration: ShapeDecoration(
          color: Colors.white,
          shape: RoundedRectangleBorder(
            side: const BorderSide(width: 0.20, color: Color(0xFF8E8E8E)),
            borderRadius: BorderRadius.circular(19),
          ),
          // borderRadius: BorderRadius.circular(8.0),
          // border: Border.all(color: Colors.grey),
          shadows: const [
            BoxShadow(
              color: Color(0x3F000000),
              blurRadius: 9.80,
              offset: Offset(0, 6),
              spreadRadius: -9,
            )
          ],
        ),
        child: Stack(
          children: [
            Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Container(
                        margin:
                            const EdgeInsets.only(left: 3, right: 3, top: 6),
                        decoration: BoxDecoration(
                          color: Colors.grey[100],
                          borderRadius: BorderRadius.circular(8.0),
                          border: Border.all(color: Colors.grey[200]!),
                        ),
                        padding: const EdgeInsets.only(left: 12),
                        child: DropdownButton<String>(
                          value: selectedFormat,
                          hint: const Text(
                            "Select Format",
                            style: TextStyle(color: Colors.black38),
                          ),
                          isExpanded: true,
                          underline: const SizedBox(),
                          items: formats.map((String value) {
                            return DropdownMenuItem<String>(
                              value: value,
                              child: Text(value),
                            );
                          }).toList(),
                          onChanged: (value) {
                            setState(() {
                              selectedFormat = value;
                            });
                            widget.onFormatSelect(value ?? '');
                          },
                        ),
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: InkWell(
                        onTap: () {
                          DateTime parsedDate;

                          try {
                            parsedDate = selectedDate == null
                                ? DateTime.now()
                                : DateFormat('d/M/yyyy').parse(selectedDate!);
                          } catch (e) {
                            parsedDate = DateTime.now();
                          }

                          _selectDate(context, parsedDate);
                        },
                        child: Container(
                          padding: const EdgeInsets.symmetric(
                              vertical: 12.0, horizontal: 16.0),
                          decoration: BoxDecoration(
                            color: Colors.grey[100],
                            borderRadius: BorderRadius.circular(8.0),
                            border: Border.all(color: Colors.grey[200]!),
                          ),
                          child: Text(
                            selectedDate == null
                                ? "Choose Date"
                                : selectedDate ?? '',
                            style: TextStyle(
                                color: selectedDate == null
                                    ? Colors.black38
                                    : Colors.black87,
                                fontSize: 16),
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: TeamSearchAutocomplete(
                          hintText: "Team A",
                          value: teamA,
                          onChanged: (value) {
                            widget.onTeamAChange(value!);
                            setState(() {
                              teamA = value;
                            });
                          }),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: TeamSearchAutocomplete(
                          hintText: "Team B",
                          value: teamB,
                          onChanged: (value) {
                            widget.onTeamBChange(value!);
                            setState(() {
                              teamB = value;
                            });
                          }),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                ElevatedButton(
                  style: selectedFormat == null ||
                          selectedDate == null ||
                          teamA == null ||
                          teamB == null
                      ? const ButtonStyle(
                          padding: WidgetStatePropertyAll(
                              EdgeInsets.symmetric(horizontal: 40)))
                      : const ButtonStyle(
                          foregroundColor: WidgetStatePropertyAll(Colors.white),
                          backgroundColor:
                              WidgetStatePropertyAll(Color(0xff0A6DEA)),
                          padding: WidgetStatePropertyAll(
                              EdgeInsets.symmetric(horizontal: 40))),
                  onPressed: selectedFormat == null ||
                          selectedDate == null ||
                          teamA == null ||
                          teamB == null
                      ? null
                      : () {
                          widget.onNextPressed(teamA!, teamB!);
                        },
                  child: const Text("NEXT"),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
