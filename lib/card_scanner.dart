import 'dart:async';

import 'package:card_scanner/models/card_details.dart';
import 'package:card_scanner/models/card_scan_options.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

export 'package:card_scanner/models/card_details.dart';
export 'package:card_scanner/models/card_issuer.dart';
export 'package:card_scanner/models/card_scan_options.dart';

class CardScanner {
  static const MethodChannel _channel = MethodChannel('basys/card_scanner');
  static const _scanCard = 'scan_card';

  static Future<CardDetails?> scanCard({CardScanOptions? scanOptions}) async {
    scanOptions ??= const CardScanOptions();
    final scanResult = await _channel.invokeMapMethod<String, String>(_scanCard, scanOptions.map);
    if (kDebugMode) {
      print("method channel : GOT VALUE FROM METHOD CHANNEL : $scanResult");
    }

    if (scanResult != null) return CardDetails.fromMap(scanResult);

    return null;
  }
}
