import 'package:flutter/material.dart';
import 'package:ossw4_msps/main.dart';

class ProductionCompanySelector
    extends StatefulWidget {
  final Function(List<String>) onChanged;

  const ProductionCompanySelector({
    super.key,
    required this.onChanged,
  });

  @override
  State<ProductionCompanySelector>
  createState() =>
      _ProductionCompanySelectorState();
}

class _ProductionCompanySelectorState
    extends State<ProductionCompanySelector> {
  final List<String> companies = [
    'Amblin Entertainment',
    'American Zoetrope',
    'ARTE',
    'ARTE France Cinéma',
    'Atresmedia',
    'BBC Film',
    'BFI',
    'Blumhouse Productions',
    'Canal+',
    'Canal+ España',
    'Castle Rock Entertainment',
    'China Film Group Corporation',
    'CJ Entertainment',
    'Ciné+',
    'Columbia Pictures',
    'Constantin Film',
    'Davis Entertainment',
    'Dimension Films',
    'DreamWorks Pictures',
    'Dune Entertainment',
    'EuropaCorp',
    'Film i Väst',
    'Film4 Productions',
    'FilmNation Entertainment',
    'Focus Features',
    'Fox 2000 Pictures',
    'Fox Searchlight Pictures',
    'France 2 Cinéma',
    'France 3 Cinéma',
    'Fís Éireann/Screen Ireland',
    'Gaumont',
    'Golan-Globus Productions',
    'Hollywood Pictures',
    'Icon Productions',
    'Imagine Entertainment',
    'INCAA',
    'Ingenious Media',
    'KADOKAWA',
    'Lakeshore Entertainment',
    'Legendary Pictures',
    'Lionsgate',
    'Malpaso Productions',
    'Medusa Film',
    'Metro-Goldwyn-Mayer',
    'MiC',
    'Millennium Media',
    'Miramax',
    'M6 Films',
    'Morgan Creek Entertainment',
    'New Line Cinema',
    'New Regency Pictures',
    'Nippon Television Network Corporation',
    'Original Film',
    'Orion Pictures',
    'Paramount Pictures',
    'Participant',
    'Pathé',
    'PolyGram Filmed Entertainment',
    'RAI',
    'RAI Cinema',
    'Regency Enterprises',
    'Relativity Media',
    'Reliance Entertainment',
    'RKO Radio Pictures',
    'Scott Free Productions',
    'Scott Rudin Productions',
    'Screen Australia',
    'Screen Gems',
    'Shin-Ei Animation',
    'Shogakukan',
    'Silver Pictures',
    'StudioCanal',
    'Summit Entertainment',
    'SVT',
    'Telefilm Canada',
    'TF1 Films Production',
    'The Cannon Group',
    'The Weinstein Company',
    'TOHO',
    'Toei Animation',
    'Toei Company',
    'Touchstone Pictures',
    'TriStar Pictures',
    'TSG Entertainment',
    'TVE',
    'uMedia',
    'UK Film Council',
    'Universal Pictures',
    'United Artists',
    'Voltage Pictures',
    'Village Roadshow Pictures',
    'Warner Bros. Pictures',
    'Walt Disney Pictures',
    'Walt Disney Productions',
    'Wild Bunch',
    'Working Title Films',
    'ZDF',
  ];

  final TextEditingController _controller =
      TextEditingController();
  final FocusNode _focusNode = FocusNode();

  List<String> selectedCompanies = [];
  List<String> filteredCompanies = [];

  @override
  void initState() {
    super.initState();
    _controller.addListener(_filterCompanies);
  }

  void _filterCompanies() {
    final input = _controller.text.toLowerCase();
    setState(() {
      filteredCompanies =
          companies
              .where(
                (c) =>
                    c.toLowerCase().contains(
                      input,
                    ) &&
                    !selectedCompanies.contains(
                      c,
                    ),
              )
              .toList();
    });
  }

  void _addCompany(String company) {
    setState(() {
      selectedCompanies.add(company);
      _controller.clear();
      filteredCompanies = [];
    });
    widget.onChanged(selectedCompanies);
  }

  void _removeCompany(String company) {
    setState(() {
      selectedCompanies.remove(company);
    });
    widget.onChanged(selectedCompanies);
  }

  void _clearAll() {
    setState(() {
      selectedCompanies.clear();
    });
    widget.onChanged([]);
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            const Text(
              "제작사",
              style: subtitleText,
            ),
            const Spacer(),
            ElevatedButton(
              onPressed: _clearAll,
              child: const Text("초기화"),
            ),
          ],
        ),
        const SizedBox(height: 8),
        TextField(
          controller: _controller,
          focusNode: _focusNode,
          decoration: const InputDecoration(
            labelText: '제작사 검색',
            border: OutlineInputBorder(),
          ),
        ),
        if (filteredCompanies.isNotEmpty)
          Container(
            margin: const EdgeInsets.only(top: 4),
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              border: Border.all(
                color: Colors.black12,
              ),
              borderRadius: BorderRadius.circular(
                4,
              ),
              color: Colors.grey[100],
            ),
            child: Column(
              children:
                  filteredCompanies.map((
                    company,
                  ) {
                    return ListTile(
                      title: Text(company),
                      onTap:
                          () => _addCompany(
                            company,
                          ),
                      dense: true,
                    );
                  }).toList(),
            ),
          ),
        const SizedBox(height: 12),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children:
              selectedCompanies.map((company) {
                return Chip(
                  label: Text(company),
                  deleteIcon: const Icon(
                    Icons.close,
                  ),
                  deleteIconColor:
                      Colors.grey[600],
                  onDeleted:
                      () =>
                          _removeCompany(company),
                );
              }).toList(),
        ),
      ],
    );
  }
}
