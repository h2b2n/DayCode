pragma solidity >=0.7.0 <=0.9.0;

contract Practice {
    uint256 private totalSupply = 10;
    string public name = "KalyLion";

    address public owner;

    constructor() public {
        owner = msg.sender;
    }

    mapping(uint256 => string) public tokenURIs;

    function getTotalSupply() public view returns (uint256) {
        return totalSupply + 100000;
    }

    function setTotalSupply(uint256 newSupply) public {
        require(owner == msg.sender, "Error");
        totalSupply = newSupply;
    }

    function setTokenUrl(uint256 id, string memory uri) public {
        tokenURIs[id] = uri;
    }
}
