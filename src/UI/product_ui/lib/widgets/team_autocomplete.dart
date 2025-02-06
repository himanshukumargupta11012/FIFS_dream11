import 'package:flutter/material.dart';
import 'package:cricgenius/services/api_services.dart';

class TeamSearchAutocomplete extends StatefulWidget {
  final String hintText;
  final String? value;
  final ValueChanged<String?> onChanged;

  const TeamSearchAutocomplete({
    super.key,
    required this.hintText,
    required this.value,
    required this.onChanged,
  });

  @override
  _TeamSearchAutocompleteState createState() => _TeamSearchAutocompleteState();
}

class _TeamSearchAutocompleteState extends State<TeamSearchAutocomplete> {
  final TextEditingController _searchController = TextEditingController();
  final FocusNode _focusNode = FocusNode();
  List<String> teams = [];
  bool _isLoading = false;
  bool _selectedFromSuggestions = false;

  OverlayEntry? _overlayEntry;
  final LayerLink _layerLink = LayerLink();

  // Define the maximum height for the suggestion list
  final double _maxSuggestionHeight = 200.0;

  @override
  void initState() {
    super.initState();
    _searchController.text = widget.value ?? '';
    _searchController.addListener(_onSearchChanged);
    _focusNode.addListener(_onFocusChanged);
  }

  @override
  void dispose() {
    _removeOverlay();
    _searchController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _onFocusChanged() {
    if (_focusNode.hasFocus && teams.isNotEmpty) {
      _showOverlay();
    } else {
      _removeOverlay();
      if (!_selectedFromSuggestions) {
        _searchController.clear();
        widget.onChanged(null);
      }
    }
  }

  Future<void> _onSearchChanged() async {
    _selectedFromSuggestions = false;
    final query = _searchController.text.trim();
    if (query.isEmpty) {
      setState(() {
        teams = [];
      });
      _removeOverlay();
      return;
    }

    setState(() {
      _isLoading = true;
    });

    final res = await ApiServices().searchTeams(query);

    setState(() {
      teams = res;
      _isLoading = false;
    });

    if (teams.isNotEmpty && _focusNode.hasFocus) {
      _showOverlay();
    } else {
      _removeOverlay();
    }
  }

  void _onSuggestionSelected(String suggestion) {
    _searchController.text = suggestion;
    _selectedFromSuggestions = true;
    widget.onChanged(suggestion);
    setState(() {
      teams = [];
    });
    _removeOverlay();
    FocusScope.of(context).unfocus();
  }

  void _clearText() {
    _searchController.clear();
    widget.onChanged('');
    setState(() {
      teams = [];
      _selectedFromSuggestions = false;
    });
    _removeOverlay();
  }

  void _showOverlay() {
    if (_overlayEntry != null) {
      _overlayEntry!.markNeedsBuild();
      return;
    }
    _overlayEntry = _createOverlayEntry();
    Overlay.of(context).insert(_overlayEntry!);
  }

  void _removeOverlay() {
    _overlayEntry?.remove();
    _overlayEntry = null;
  }

  OverlayEntry _createOverlayEntry() {
    RenderBox renderBox = context.findRenderObject() as RenderBox;
    Size size = renderBox.size;
    Offset position = renderBox.localToGlobal(Offset.zero);

    return OverlayEntry(
      builder: (context) => Positioned(
        width: size.width,
        child: CompositedTransformFollower(
          link: _layerLink,
          showWhenUnlinked: false,
          offset: Offset(0.0, size.height + 5.0),
          child: Material(
            elevation: 4.0,
            borderRadius: BorderRadius.circular(8),
            child: ConstrainedBox(
              constraints: BoxConstraints(
                maxHeight: _maxSuggestionHeight,
              ),
              child: ListView.builder(
                padding: EdgeInsets.zero,
                shrinkWrap: true,
                itemCount: teams.length,
                itemBuilder: (context, index) {
                  final team = teams[index];
                  return InkWell(
                    onTap: () => _onSuggestionSelected(team),
                    child: Container(
                      padding:
                          const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                      child: Text(
                        team,
                        style: const TextStyle(
                          color: Colors.black87,
                          fontSize: 16,
                        ),
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    const textFieldStyle = TextStyle(
      color: Colors.black87,
      fontSize: 16,
    );

    const hintStyle = TextStyle(
      color: Colors.black38,
      fontSize: 16,
    );

    return CompositedTransformTarget(
      link: _layerLink,
      child: TextField(
        controller: _searchController,
        focusNode: _focusNode,
        style: textFieldStyle,
        decoration: InputDecoration(
          hintText: widget.hintText,
          hintStyle: hintStyle,
          contentPadding:
              const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
          filled: true,
          fillColor: Colors.grey[100],
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
            borderSide: BorderSide(color: Colors.grey[200]!),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(8),
            borderSide: const BorderSide(color: Colors.blueAccent),
          ),
          suffixIcon: _isLoading
              ? const Padding(
                  padding: EdgeInsets.all(10.0),
                  child: SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2.0,
                      color: Colors.blueAccent,
                    ),
                  ),
                )
              : _searchController.text.isNotEmpty
                  ? IconButton(
                      icon: Icon(Icons.clear, color: Colors.grey[600]),
                      onPressed: _clearText,
                    )
                  : null,
        ),
      ),
    );
  }
}
