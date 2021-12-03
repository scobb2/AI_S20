'use strict';
let readParams = require("./ReadParams.js");
let guessUtils = require("./GuessUtils.js");
let rls = require('readline-sync');

let mMind = function() {
   let numGames = 0, totalTries = 0, guess, matches, model = [];
   let params = readParams();
   let average;
   
   console.log("Params: ", params);
   while (true) {
      average = totalTries / numGames;
      console.log("Current average: ",
       isNaN(average) ? "N/A" : average.toPrecision(2));
      if (rls.question('Play a game? ').trim().toUpperCase().charAt(0) !== "Y")
         break;

      numGames++;
      for (let i = 0; i < params.length; i++) {
         model[i] = String.fromCharCode('A'.charCodeAt(0)
          + Math.random() * params.randRange);
      }
      console.log("Here's the answer: ", model.join(''));
      do {
         matches = guessUtils.findMatches(model, guessUtils.getGuess(params));
         console.log(`${matches.exact} exact and ${matches.inexact} inexact.`);
         totalTries++;
      } while (matches.exact < params["length"]);
   }
};

mMind();
