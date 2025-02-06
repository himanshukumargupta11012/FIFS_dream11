import 'package:cricgenius/screen_router.dart';
import 'package:cricgenius/screens/home.dart';
import 'package:cricgenius/services/api_services.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:intl/date_symbol_data_local.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await initializeDateFormatting();

  final apiServices = ApiServices();
  await apiServices.configureDio();
  runApp(const MyApp());
}

class MyApp extends StatefulWidget {
  const MyApp({super.key});

  @override
  State<MyApp> createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: "CricGenius",
      theme: ThemeData(appBarTheme: const AppBarTheme(systemOverlayStyle: SystemUiOverlayStyle.light)),
      debugShowCheckedModeBanner: false,
      routes: {
        '/': (context) => const ScreenRouter(),
        '/home': (context) => const HomeScreen(),
      },
    );
  }
}
