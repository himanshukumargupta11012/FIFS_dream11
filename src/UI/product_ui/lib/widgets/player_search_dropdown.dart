import 'package:cricgenius/model/player.dart';
import 'package:cricgenius/services/api_services.dart';
import 'package:flutter/material.dart';
import 'package:dio/dio.dart';

class CustomSearchAutocomplete extends StatefulWidget {
  final String hintText;
  final String? value;
  final ValueChanged<String?> onChanged;

  const CustomSearchAutocomplete({
    super.key,
    required this.hintText,
    required this.value,
    required this.onChanged,
  });

  @override
  _CustomSearchAutocompleteState createState() =>
      _CustomSearchAutocompleteState();
}

class _CustomSearchAutocompleteState extends State<CustomSearchAutocomplete> {
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _focusNode = FocusNode();
  final Dio dio = Dio();
  List<PlayerModel> _players = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _searchController.text = widget.value ?? '';
    _searchController.addListener(_onSearchChanged);
    _focusNode.addListener(_onFocusChanged);
  }

  @override
  void dispose() {
    _searchController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _onFocusChanged() {
    setState(() {
      // Rebuild to show/hide suggestions when focus changes
    });
  }

  Future<void> _onSearchChanged() async {
    final query = _searchController.text.trim();
    if (query.isEmpty) {
      setState(() {
        _players = [];
      });
      return;
    }

    setState(() {
      _isLoading = true;
    });

    final res = await ApiServices().searchPlayers(query);

    setState(() {
      _players = res;
      _isLoading = false;
    });

  }

  void _onSuggestionSelected(String suggestion) {
    _searchController.text = suggestion;
    widget.onChanged(suggestion);
    setState(() {
      _players = [];
    });
    // Optionally, unfocus the TextField
    FocusScope.of(context).unfocus();
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        TextField(
          controller: _searchController,
          focusNode: _focusNode,
          decoration: InputDecoration(
            hintText: widget.hintText,
            suffixIcon: _isLoading
                ? const Padding(
                    padding: EdgeInsets.all(10.0),
                    child: SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2.0),
                    ),
                  )
                : null,
          ),
        ),
        if (_players.isNotEmpty && _focusNode.hasFocus)
          Container(
            constraints: const BoxConstraints(
              maxHeight: 200,
            ),
            child: ListView.builder(
              shrinkWrap: true,
              itemCount: _players.length,
              itemBuilder: (context, index) {
                final player = _players[index];
                return ListTile(
                  title: Text(player.name),
                  onTap: () => _onSuggestionSelected(player.name),
                );
              },
            ),
          ),
      ],
    );
  }
}
