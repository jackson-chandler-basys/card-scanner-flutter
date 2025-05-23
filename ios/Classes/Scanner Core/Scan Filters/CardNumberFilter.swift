//
//  CardNumberFilter.swift
//  card_scanner
//
//  Created by Abhinav Kumar on 12/02/21.
//

import Foundation
import MLKitTextRecognition

class CardNumberFilter: ScanFilter {
    private var cardNumberRegex: NSRegularExpression = try! NSRegularExpression(pattern: CardScannerRegexps.cardNumberRegex, options: [.anchorsMatchLines])
    
    var visionText: Text
    var scannerOptions: CardScannerOptions
    
    init(visionText: Text, scannerOptions: CardScannerOptions) {
        self.visionText = visionText
        self.scannerOptions = scannerOptions
    }
    
    func filter() -> ScanFilterResult? {
        for (blockIndex, block) in visionText.blocks.enumerated() {
            for (_, line) in block.lines.enumerated() {
                let transformedBlockText = transformBlockText(blockText: line.text)
                let sanitizedBlockText = transformedBlockText.sanitized
                debugLog("Sanitized Card Number : \(sanitizedBlockText)", scannerOptions: scannerOptions)
                
                if let firstMatch = cardNumberRegex.firstMatch(
                    in: sanitizedBlockText,
                    range: NSRange(location: 0, length: sanitizedBlockText.count)
                ) {
                    let cardNumber = (sanitizedBlockText as NSString).substring(with: firstMatch.range).trimmingCharacters(in: .whitespacesAndNewlines) as String
                    
                    if cardNumber.isNotValidCardNumber {
                        continue
                    }
                    
                    debugLog("Card Number : \(cardNumber)", scannerOptions: scannerOptions)
                    
                    if scannerOptions.enableLuhnCheck && cardNumber.isNotValidCardNumber {
                        debugLog("Luhn check failed for card number (\(cardNumber))", scannerOptions: scannerOptions)
                        continue
                    }
                    
                    return CardNumberScanResult(
                        visionText: visionText,
                        textBlockIndex: blockIndex,
                        textBlock: block,
                        cardNumber: cardNumber
                    )
                }
            }
        }
        
        return nil
    }
    func transformBlockText(blockText: String) -> String {
        return blockText.replacingOccurrences(of: "c", with: "C")
            .replacingOccurrences(of: "o", with: "O")
            .replacingOccurrences(of: "b", with: "6")
            .replacingOccurrences(of: "+", with: "4")
            .replacingOccurrences(of: "L", with: "1")
            .replacingOccurrences(of: "Y", with: "4")
            .replacingOccurrences(of: "I", with: "1")
            .replacingOccurrences(of: "S", with: "5")
            .replacingOccurrences(of: "D", with: "0")

    }
    
}

extension String {
    var isValidCardNumber: Bool {
        return true
    }
    
    var isNotValidCardNumber: Bool {
        return !isValidCardNumber
    }
    
    var isLuhnValidCardNumber: Bool {
        let digitList: [Int] = self.reversed().enumerated().map { (index, element) in
            var num = Int("\(element)") ?? 0
            if (index % 2 == 1) {
                num = (num * 2)
                num = (num == 0) ? num : (num % 9 == 0) ? 9 : num % 9
            }
            return num
        }
        
        return (digitList.reduce(0, +)) % 10 == 0
    }
    
    var isNotLuhnValidCardNumber: Bool {
        return !isLuhnValidCardNumber
    }
}
